import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple, Dict, List, Any
from tqdm.auto import tqdm
import os
import torch

QA_MAX_ANSWER_LENGTH = 30


def qa_data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom data collator for QA that ensures start_positions and end_positions are preserved.
    """
    # Get all keys from the first feature
    if not features:
        return {}

    first = features[0]
    batch = {}

    # For each key in the features, stack them into a batch
    for key in first.keys():
        if key in [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "start_positions",
            "end_positions",
        ]:
            # These should be tensors
            batch[key] = torch.tensor([f[key] for f in features])
        elif key in ["offset_mapping", "example_id"]:
            # These are metadata, keep as list
            batch[key] = [f[key] for f in features]
        else:
            # For any other key, try to tensorize
            try:
                batch[key] = torch.tensor([f[key] for f in features])
            except (ValueError, TypeError):
                # If it fails, keep as list
                batch[key] = [f[key] for f in features]

    return batch


def adjust_output_dir(out_dir: str) -> str:
    """Normalize and place output directories under a top-level ./Data folder.

    Behavior:
    - If the provided path (after normalization) already starts with 'Data', it is returned unchanged.
    - If an absolute path is provided, only the basename is used and placed under './Data/'.
      e.g. '/abs/path/model' -> 'Data/model'
    - For relative paths, prepend 'Data/'.

    This mirrors the small helper previously embedded in `eval.py` so both train/eval
    scripts can share the same logic.
    """
    norm = os.path.normpath(out_dir)
    parts = norm.split(os.sep)
    # If already starts with Data (relative), keep as-is
    if parts and parts[0] == "Data":
        return norm
    # If absolute path provided, use only the basename under ./Data
    if os.path.isabs(out_dir):
        return os.path.join("Data", os.path.basename(out_dir))
    # Normal relative path: prepend Data/
    return os.path.join("Data", out_dir)


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Calculate start_positions and end_positions for loss computation
        # (Logic adapted from prepare_train_dataset_qa)
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        offsets = tokenized_examples["offset_mapping"][i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != context_index:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != context_index:
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        )
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features."
        )

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def evaluate(
        self,
        eval_dataset=None,  # denotes the dataset after mapping
        eval_examples=None,  # denotes the raw dataset
        ignore_keys=None,  # keys to be ignored in dataset
        metric_key_prefix: str = "eval",
    ):
        """
        NUCLEAR OPTION: Completely custom evaluation loop.
        We bypass Trainer's evaluation_loop entirely and do everything ourselves.
        """
        import torch
        import numpy as np
        from tqdm import tqdm

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Set model to eval mode
        self.model.eval()

        # Initialize storage for predictions and losses
        all_start_logits = []
        all_end_logits = []
        all_losses = []

        # Manual evaluation loop
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = self._prepare_inputs(batch)

            with torch.no_grad():
                # Forward pass
                outputs = self.model(**batch)

                # Extract loss
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    all_losses.append(outputs.loss.item())

                # Extract logits for QA metrics
                all_start_logits.append(outputs.start_logits.cpu().numpy())
                all_end_logits.append(outputs.end_logits.cpu().numpy())

        # Concatenate all predictions
        all_start_logits = np.concatenate(all_start_logits, axis=0)
        all_end_logits = np.concatenate(all_end_logits, axis=0)
        predictions = (all_start_logits, all_end_logits)

        # Initialize metrics dictionary
        metrics = {}

        # 1. COMPUTE LOSS - This is what we've been fighting for!
        if all_losses:
            eval_loss = np.mean(all_losses)
            metrics[f"{metric_key_prefix}_loss"] = eval_loss
            print(f"✅ Computed {metric_key_prefix}_loss: {eval_loss:.4f}")
        else:
            print(f"⚠️ No losses computed!")

        # 2. COMPUTE QA METRICS (F1, Exact Match)
        if self.compute_metrics is not None and eval_examples is not None:
            # Post-process predictions (convert logits to answer strings)
            eval_preds = postprocess_qa_predictions(
                eval_examples, eval_dataset, predictions
            )

            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in eval_preds.items()
            ]
            references = [
                {"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples
            ]

            # Compute F1 and EM metrics
            computed_metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions, label_ids=references)
            )

            # Add to metrics dict
            for key, value in computed_metrics.items():
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = value
                else:
                    metrics[key] = value

        # 3. Add other standard metrics
        metrics[f"{metric_key_prefix}_samples"] = len(eval_dataset)

        # DIRECTLY log to state.log_history
        self.state.log_history.append({**metrics, "step": self.state.global_step})

        # Also use Trainer's log method (though it might not work as expected)
        self.log(metrics)

        # Trigger callbacks
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics
