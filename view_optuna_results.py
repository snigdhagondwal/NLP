import optuna
import pandas as pd
import os
import glob

def view_optuna_results(study_name=None, db_path=None):
    """View detailed results from Optuna hyperparameter optimization"""
    
    # Find Optuna database file if not specified
    if db_path is None:
        db_files = glob.glob("optuna_*.db")
        if not db_files:
            print("❌ No Optuna database files found!")
            return
        db_path = db_files[0]  # Use the most recent one
        print(f"📂 Using database: {db_path}")
    
    # Extract study name from database filename if not provided
    if study_name is None:
        study_name = os.path.basename(db_path).replace("optuna_", "").replace(".db", "")
    
    try:
        # Load the study
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}"
        )
        
        print(f"\n🎯 OPTUNA STUDY RESULTS: {study_name}")
        print("=" * 60)
        
        # Overall best results
        print(f"🏆 Best Trial: #{study.best_trial.number}")
        print(f"🎯 Best F1 Score: {-study.best_value:.4f}")
        print(f"📊 Total Trials: {len(study.trials)}")
        print(f"✅ Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        print(f"\n🔧 BEST HYPERPARAMETERS:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Detailed trial results
        print(f"\n📋 DETAILED TRIAL RESULTS:")
        print("-" * 80)
        
        trials_data = []
        for trial in study.trials:
            trial_info = {
                'Trial': trial.number,
                'State': trial.state.name,
                'F1_Score': f"{-trial.value:.4f}" if trial.value is not None else "N/A",
                'Duration': f"{trial.duration.total_seconds():.0f}s" if trial.duration else "N/A"
            }
            
            # Add hyperparameters
            for param, value in trial.params.items():
                trial_info[param] = value
            
            trials_data.append(trial_info)
        
        # Create DataFrame for better display
        df = pd.DataFrame(trials_data)
        print(df.to_string(index=False))
        
        # Save to CSV for future reference
        csv_file = f"optuna_results_{study_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved to: {csv_file}")
        
        return study, df
        
    except Exception as e:
        print(f"❌ Error loading study: {e}")
        return None, None

def compare_trials(study):
    """Compare trial performance"""
    if study is None:
        return
    
    print(f"\n📊 TRIAL PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if completed_trials:
        f1_scores = [-t.value for t in completed_trials]
        print(f"📈 Best F1: {max(f1_scores):.4f}")
        print(f"📉 Worst F1: {min(f1_scores):.4f}")
        print(f"📊 Average F1: {sum(f1_scores)/len(f1_scores):.4f}")
        
        # Show top 3 trials
        sorted_trials = sorted(completed_trials, key=lambda x: -x.value)
        print(f"\n🥇 TOP 3 TRIALS:")
        for i, trial in enumerate(sorted_trials[:3]):
            print(f"  {i+1}. Trial #{trial.number}: F1={-trial.value:.4f}")
            for param, value in trial.params.items():
                print(f"     {param}: {value}")
            print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View Optuna hyperparameter optimization results")
    parser.add_argument("--study_name", type=str, help="Name of the Optuna study")
    parser.add_argument("--db_path", type=str, help="Path to Optuna database file")
    
    args = parser.parse_args()
    
    # View results
    study, df = view_optuna_results(args.study_name, args.db_path)
    
    if study:
        compare_trials(study)