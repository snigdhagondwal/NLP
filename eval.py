import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    HfArgumentParser,
)
import evaluate
from helpers import (
    prepare_validation_dataset_qa,
    QuestionAnsweringTrainer,
    adjust_output_dir,
)
import os
import json
from typing import Optional, Dict, Any

NUM_PREPROCESSING_WORKERS = 2


def evaluate_qa_model(
    model_path: str,
    dataset: str = "squad",
    output_dir: str = "eval_output",
    max_length: int = 128,
    max_eval_samples: Optional[int] = None,
    per_device_eval_batch_size: int = 8,
    save_predictions: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate a Question Answering model.

    Args:
        model_path: Path to the trained model checkpoint or HuggingFace model ID
        dataset: Dataset to evaluate on ('squad' or path to JSON/JSONL file)
        output_dir: Where to save evaluation results
        max_length: Maximum sequence length for evaluation
        max_eval_samples: Limit the number of evaluation examples (None for all)
        per_device_eval_batch_size: Evaluation batch size
        save_predictions: Whether to save predictions to file
        **kwargs: Additional TrainingArguments parameters

    Returns:
        results: Dictionary containing evaluation metrics
    """
    # Ensure outputs go under ./Data using shared helper in helpers.py
    # output_dir = adjust_output_dir(output_dir)

    # Load dataset
    if dataset.endswith(".json") or dataset.endswith(".jsonl"):
        # Load from local json/jsonl file
        dataset_obj = datasets.load_dataset("json", data_files=dataset)
        # The "json" loader places all examples in train split by default
        eval_split = "train"
    else:
        # Load SQuAD or other dataset from HuggingFace
        dataset_obj = datasets.load_dataset(dataset)
        eval_split = "validation"

    # Load model and tokenizer from checkpoint
    print(f"✅ Loading model from {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Prepare dataset preprocessing function
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)

    print(
        "✅ Preprocessing evaluation data... (this takes a little bit, should only happen once per dataset)"
    )

    eval_dataset = dataset_obj[eval_split]
    if max_eval_samples:
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    eval_dataset_featurized = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names,
    )

    # Setup evaluation metrics
    metric = evaluate.load("squad")
    compute_metrics = lambda eval_preds: metric.compute(
        predictions=eval_preds.predictions, references=eval_preds.label_ids
    )

    # Store predictions for later dumping
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Create TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        **kwargs,
    )

    # Initialize the QuestionAnsweringTrainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
    )

    # Evaluate the model
    print("✅ Starting evaluation...")
    results = trainer.evaluate(eval_examples=eval_dataset)

    print("\n" + "=" * 50)
    print("✅ Evaluation Results:")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    if save_predictions:
        # Save evaluation metrics
        os.makedirs(output_dir, exist_ok=True)

        metrics_path = os.path.join(output_dir, "eval_metrics.json")
        with open(metrics_path, encoding="utf-8", mode="w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Metrics saved to {metrics_path}")

        # Save predictions
        predictions_path = os.path.join(output_dir, "eval_predictions.jsonl")
        with open(predictions_path, encoding="utf-8", mode="w") as f:
            predictions_by_id = {
                pred["id"]: pred["prediction_text"]
                for pred in eval_predictions.predictions
            }
            for example in eval_dataset:
                example_with_prediction = dict(example)
                example_with_prediction["predicted_answer"] = predictions_by_id[
                    example["id"]
                ]
                f.write(json.dumps(example_with_prediction))
                f.write("\n")
        print(f"✅ Predictions saved to {predictions_path}")

    print("\n☑️ Evaluation complete!")

    return results


def main():
    argp = HfArgumentParser(TrainingArguments)
    # Key TrainingArguments to specify when calling from command line:
    # --output_dir <path>: Where to save evaluation results (required)
    # --per_device_eval_batch_size <int, default=8>: Evaluation batch size

    argp.add_argument(
        "--model",
        type=str,
        required=True,
        help="""Path to the trained model checkpoint or HuggingFace model ID 
                      to evaluate.""",
    )
    argp.add_argument(
        "--dataset",
        type=str,
        default="squad",
        help="""Dataset to evaluate on. Default is 'squad'. Can also specify 
                      a custom JSON/JSONL file path.""",
    )
    argp.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="""Maximum sequence length for evaluation. 
                      Should match the length used during training.""",
    )
    argp.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit the number of evaluation examples.",
    )

    training_args, args = argp.parse_args_into_dataclasses()

    # Call the functional evaluate_qa_model
    evaluate_qa_model(
        model_path=args.model,
        dataset=args.dataset,
        output_dir=training_args.output_dir,
        max_length=args.max_length,
        max_eval_samples=args.max_eval_samples,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    )


if __name__ == "__main__":
    main()
