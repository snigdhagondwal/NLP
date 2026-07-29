import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    HfArgumentParser,
)
from helpers import (
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
    QuestionAnsweringTrainer,
    adjust_output_dir,
    qa_data_collator,
)
import argparse
import evaluate
from typing import Optional

NUM_PREPROCESSING_WORKERS = 2


def train_qa_model(
    model_name: str = "google/electra-small-discriminator",
    dataset: str = "squad",
    validation_dataset: Optional[str] = None,
    output_dir: str = "model_output",
    max_length: int = 128,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    num_train_epochs: float = 3.0,
    learning_rate: float = 5e-5,
    warmup_steps: int = 0,
    logging_steps: int = 500,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    do_eval: bool = True,
    **kwargs,
):
    """
    Train a Question Answering model.

    Args:
        model_name: Base model to fine-tune (HuggingFace model ID or path)
        dataset: Dataset to use ('squad' or path to JSON/JSONL file)
        validation_dataset: Optional separate validation dataset (path to JSON/JSONL file)
        output_dir: Where to save model checkpoints
        max_length: Maximum sequence length for training
        max_train_samples: Limit the number of training examples (None for all)
        max_eval_samples: Limit the number of evaluation examples (None for all)
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        num_train_epochs: Number of passes through training data
        learning_rate: Learning rate for training
        warmup_steps: Number of warmup steps
        logging_steps: Log every X updates steps
        save_steps: Save checkpoint every X updates steps
        eval_steps: Evaluate every X steps (None uses eval_strategy)
        eval_strategy: When to evaluate ('no', 'steps', 'epoch')
        save_strategy: When to save ('no', 'steps', 'epoch')
        load_best_model_at_end: Load the best model at the end of training
        metric_for_best_model: Metric to use for best model selection ('f1' or 'exact_match')
        do_eval: Whether to run evaluation during training
        **kwargs: Additional TrainingArguments parameters

    Returns:
        trainer: The trained QuestionAnsweringTrainer object
    """
    # Ensure outputs go under ./Data using shared helper in helpers.py
    original_output_dir = output_dir
    output_dir = adjust_output_dir(output_dir)

    # Load dataset
    if dataset.endswith(".json") or dataset.endswith(".jsonl"):
        # Load from local json/jsonl file
        dataset_obj = datasets.load_dataset("json", data_files=dataset)
        train_split = "train"

        # If validation_dataset is provided, load it separately
        if validation_dataset:
            if validation_dataset.endswith(".json") or validation_dataset.endswith(
                ".jsonl"
            ):
                validation_dataset_obj = datasets.load_dataset(
                    "json", data_files=validation_dataset
                )
                eval_split = (
                    "train"  # The validation file's data is in the "train" split
                )
                use_separate_validation = True
            else:
                raise ValueError("validation_dataset must be a JSON or JSONL file")
        else:
            # Use training data for evaluation (not recommended)
            eval_split = "train"
            use_separate_validation = False
    else:
        # Load SQuAD or other dataset from HuggingFace
        dataset_obj = datasets.load_dataset(dataset)
        train_split = "train"
        eval_split = "validation"
        use_separate_validation = False

    # Initialize model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Make tensor contiguous if needed (for ELECTRA models)
    # https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Prepare dataset preprocessing functions
    prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)

    print(
        "☑️ Preprocessing training data... (this takes a little bit, should only happen once per dataset)"
    )

    train_dataset = dataset_obj[train_split]
    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    train_dataset_featurized = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=train_dataset.column_names,
    )

    # Prepare evaluation dataset if do_eval is True
    eval_dataset_featurized = None
    eval_dataset_examples = None
    compute_metrics_fn = None

    if do_eval:
        print("☑️ Preprocessing evaluation data...")

        # Use separate validation dataset if provided, otherwise use the same dataset object
        if use_separate_validation:
            eval_dataset = validation_dataset_obj[eval_split]
        else:
            eval_dataset = dataset_obj[eval_split]

        if max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset_examples = eval_dataset  # Store for compute_metrics

        # Get original column names before mapping
        original_columns = eval_dataset.column_names

        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            # Remove original columns but keep the new ones added by prepare_eval_dataset
            # (including start_positions and end_positions needed for loss computation)
            remove_columns=original_columns,
        )

        # Setup evaluation metrics
        metric = evaluate.load("squad")
        compute_metrics_fn = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions,
            references=eval_preds.label_ids,
        )

    # Create TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy=eval_strategy if do_eval else "no",
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end if do_eval else False,
        metric_for_best_model=f"eval_{metric_for_best_model}" if do_eval else None,
        greater_is_better=True if do_eval else None,
        label_names=["start_positions", "end_positions"],
        **kwargs,
    )

    # Initialize the QuestionAnsweringTrainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        eval_examples=eval_dataset_examples,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        data_collator=qa_data_collator,  # Use custom collator that preserves labels
    )

    # Train the model
    print("✅ Starting training...")
    trainer.train()

    # Save the final model
    print(f"✅ Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("✅ Training complete!")

    return trainer


def main():
    argp = HfArgumentParser(TrainingArguments)
    # Key TrainingArguments to specify when calling from command line:
    # --per_device_train_batch_size <int, default=8>: Training batch size
    # --num_train_epochs <float, default=3.0>: Number of passes through training data
    # --output_dir <path>: Where to save model checkpoints (required)
    # --learning_rate <float, default=5e-5>: Learning rate for training
    # --warmup_steps <int, default=0>: Number of warmup steps
    # --logging_steps <int, default=500>: Log every X updates steps
    # --save_steps <int, default=500>: Save checkpoint every X updates steps

    argp.add_argument(
        "--model",
        type=str,
        default="google/electra-small-discriminator",
        help="""Base model to fine-tune. Should be a HuggingFace model ID 
                      (see https://huggingface.co/models) or path to a saved checkpoint.""",
    )
    argp.add_argument(
        "--dataset",
        type=str,
        default="squad",
        help="""Dataset to use. Default is 'squad'. Can also specify a custom 
                      JSON/JSONL file path.""",
    )
    argp.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="""Maximum sequence length for training. 
                      Shorter lengths use less memory but may truncate examples.""",
    )
    argp.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of training examples.",
    )

    training_args, args = argp.parse_args_into_dataclasses()

    # Call the functional train_qa_model
    train_qa_model(
        model_name=args.model,
        dataset=args.dataset,
        output_dir=training_args.output_dir,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        warmup_steps=training_args.warmup_steps,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
    )


if __name__ == "__main__":
    main()
