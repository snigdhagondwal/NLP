from train import train_qa_model
from eval import evaluate_qa_model
import os

# MLflow setup
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
#run_name
# Configuration
model_name = "./Data/electra_squad_model"  # You will need to change this
output_dir = "./Data/checklist_trained_model"

# Train ELECTRA on SQuAD
trainer = train_qa_model(
    # Model and dataset
    model_name="google/electra-base-discriminator",        
    dataset="./Datasets/combined_train.jsonl",
    validation_dataset="./Datasets/combined_validation.jsonl",
    output_dir="./Data/combined_electra_base_model",
    # Training parameters
    num_train_epochs=4,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    warmup_steps=500,
    # Evaluation parameters
    do_eval=True,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    # Model selection
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    # Logging
    logging_steps=10,
    logging_strategy="steps",
    logging_first_step=True,
    # Demo with subset of data
    # max_train_samples=50,
    # max_eval_samples=50,
    use_mps_device=True,
    report_to=["mlflow"],
)

print("\n✅ Training complete! Best model saved. View results with: mlflow ui")
