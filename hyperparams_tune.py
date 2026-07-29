from train import train_qa_model
from eval import evaluate_qa_model
import os
import optuna
import shutil
from datetime import datetime
import mlflow

# MLflow setup
os.makedirs("mlruns", exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 6)
    warmup_steps = trial.suggest_int('warmup_steps', 0, 1000)
    per_device_eval_batch_size = trial.suggest_categorical('per_device_eval_batch_size', [16, 32, 64])
    
    # Create unique output directory for this trial
    trial_output_dir = f"./Data/optuna_trial_{trial.number}"
    
    # Ensure the Data directory exists
    os.makedirs("./Data", exist_ok=True)
    
    try:
        # Train model with suggested hyperparameters
        trainer = train_qa_model(
            # Model and dataset
            dataset="./Datasets/combined_train.jsonl",
            validation_dataset="./Datasets/combined_validation.jsonl",
            output_dir=trial_output_dir,
            # Hyperparameters to optimize
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            # Fixed parameters
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            logging_strategy="steps",
            use_mps_device=True,
            report_to=[],  # Disable MLflow logging during hyperparameter search
            # Optionally limit samples for faster trials
            # max_train_samples=1000,
            # max_eval_samples=500,
        )
        
        # Get the best validation F1 score
        # The trainer logs metrics, we want to get the best eval_f1
        log_history = trainer.state.log_history
        eval_f1_scores = [log['eval_f1'] for log in log_history if 'eval_f1' in log]
        
        if eval_f1_scores:
            best_f1 = max(eval_f1_scores)
        else:
            # Fallback: run evaluation manually
            eval_results = trainer.evaluate()
            best_f1 = eval_results.get('eval_f1', 0.0)
        
        # Save detailed trial results before deletion
        trial_results = {
            'trial_number': trial.number,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'per_device_train_batch_size': per_device_train_batch_size,
                'num_train_epochs': num_train_epochs,
                'warmup_steps': warmup_steps,
                'per_device_eval_batch_size': per_device_eval_batch_size
            },
            'best_f1': best_f1,
            'all_eval_f1_scores': eval_f1_scores,
            'training_logs': log_history
        }
        
        # Save to JSON file
        import json
        os.makedirs("./Optuna/optuna_trials", exist_ok=True)
        with open(f"./Optuna/optuna_trials/trial_{trial.number}_results.json", 'w') as f:
            json.dump(trial_results, f, indent=2)
        
        # Optuna minimizes by default, so return negative F1 to maximize F1
        objective_value = -best_f1
        
        # Log trial results
        print(f"🎉🎉🎉Trial {trial.number} completed:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {per_device_train_batch_size}")
        print(f"  Epochs: {num_train_epochs}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Objective: {objective_value:.4f}")
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a high value for failed trials
        objective_value = float('inf')
    
    finally:
        # Optionally clean up trial directory to save disk space
        # Comment out the next 3 lines if you want to keep trial results
        # if os.path.exists(trial_output_dir):
        #     shutil.rmtree(trial_output_dir)
        pass  # Keep trial directories for detailed analysis
    
    return objective_value

def run_hyperparameter_search(n_trials=5, study_name=None, experiment_name="QA_Hyperparameter_Optimization"):
    """Run Optuna hyperparameter optimization"""
    
    if study_name is None:
        study_name = f"qa_model_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Ensure directories exist for Optuna database and other outputs
    os.makedirs("./Data", exist_ok=True)
    os.makedirs("./SQLITE", exist_ok=True)
    
    # Create study with pruning for early stopping of bad trials
    study = optuna.create_study(
        direction='minimize',  # Minimizing negative F1 = maximizing F1
        study_name=study_name,
        storage=f'sqlite:///./SQLITE/optuna_{study_name}.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1
        )
    )
    
    print(f"🔍 Starting hyperparameter search with {n_trials} trials...")
    print(f"📊 Study name: {study_name}")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("\n" + "="*50)
    print("🎉 HYPERPARAMETER SEARCH COMPLETE!")
    print("="*50)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 score: {-study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best hyperparameters
    print(f"\n🚀 Training final model with best hyperparameters...")
    best_params = study.best_params
    
    # Ensure final model output directory exists
    final_output_dir = "./Data/combined_models_optimized"
    os.makedirs(os.path.dirname(final_output_dir), exist_ok=True)
    
    # Start MLflow run with a descriptive name
    with mlflow.start_run(run_name=f"best_model_{study_name}"):
        # Log best hyperparameters
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_from_optuna", -study.best_value)
        
        final_trainer = train_qa_model(
            # Model and dataset
            dataset="./Datasets/combined_train.jsonl",
            validation_dataset="./Datasets/combined_validation.jsonl",
            output_dir=final_output_dir,
            # Best hyperparameters
            **best_params,
            # Fixed parameters
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            logging_strategy="steps",
            logging_first_step=True,
            use_mps_device=True,
            report_to=["mlflow"],
        )
    
    print("\n✅ Hyperparameter search and final training complete!")
    print(f"✅ Best model saved to {final_output_dir}")
    print("✅ View MLflow results with: mlflow ui")
    
    return study, final_trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for QA model")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of optimization trials")
    parser.add_argument("--study_name", type=str, help="Name for the Optuna study")
    parser.add_argument("--experiment_name", type=str, default="QA_Hyperparameter_Optimization", help="MLflow experiment name")
    
    args = parser.parse_args()
    
    # Run hyperparameter optimization
    study, trainer = run_hyperparameter_search(n_trials=args.n_trials, study_name=args.study_name, experiment_name=args.experiment_name)