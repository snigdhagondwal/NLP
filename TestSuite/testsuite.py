
'''
we have three types of analysis:

1. SQuAD Evaluation f1 & exact match. --> run it on Squad dataset, Adversarial dataset, and (optionally) Checklist dataset
2. Adversarial Test Suite Analysis --> run it on Adversarial dataset, Squad dataset, and (optionally) Checklist dataset
3. Checklist Test Suite Analysis --> run it only on checklist dataset 

'''


import os
import sys; sys.path.append("../")
import json
from eval import evaluate_qa_model
from Analyse.Checklist.helper import ChecklistTestRunner
from Analyse.Adversarial.helper import run_analysis

# model_path="./Data/trained_model"
model_path="../Models/combined_extra_epochs_model"

checklist_dataset = "../Datasets/checklist_test.jsonl"
adversarial_dataset = "../Datasets/adversarial_qa_valid.jsonl"
output_folder= "../Results/"
output_path=f"{output_folder}/{os.path.basename(model_path)}/"

metrics = {}


# run inference of the models & get the custom evaluation metrics
print(f"✅ Running inference on Squad Dataset...") 
results=evaluate_qa_model(model_path=model_path,
                  dataset="squad",
                  output_dir=f"{output_path}/squad_results")

filtered_results = {k: v for k, v in results.items() if k in ['eval_exact_match', 'eval_f1']}
metrics['squad'] = filtered_results

print(f"✅ Running inference on Adversarial Dataset...")
results=evaluate_qa_model(model_path=model_path,
                  dataset=adversarial_dataset,
                  output_dir=f"{output_path}/adversarial_results")

filtered_results = {k: v for k, v in results.items() if k in ['eval_exact_match', 'eval_f1']}
metrics['adversarial'] = filtered_results


# run adversarial test suite analysis
print(f"✅ Running Adversarial Test Suite Analysis on Squad Dataset...")
macro_analysis = run_analysis(predictions_file=f"{output_path}/squad_results/eval_predictions.jsonl",
                              output_file=f"{output_path}/adversarial_analysis/squad_dataset.md",
                              model_name=os.path.basename(model_path))
metrics['squad'].update(macro_analysis)


print(f"✅ Running Adversarial Test Suite Analysis on Adversarial Dataset...")
macro_analysis = run_analysis(predictions_file=f"{output_path}/adversarial_results/eval_predictions.jsonl",
                              output_file=f"{output_path}/adversarial_analysis/adversarial_dataset.md",
                              model_name=os.path.basename(model_path))
metrics['adversarial'].update(macro_analysis)




# run checklist test suite analysis
print(f"✅ Running Checklist Test Suite Analysis ...") 
runner = ChecklistTestRunner(model_path=model_path)
df = runner.run_suite_on_jsonl(jsonl_file=checklist_dataset, save_results=True, output_folder=f"{output_path}/checklist_analysis")

checklist_metrics= {}
for index, row in df.iterrows():
    test_name = row['Test Name']
    checklist_metrics[test_name] = {
        'failure_rate': row['Failure Rate'],
        'average_failure': row['Average Failure']
    }

metrics['checklist'] = checklist_metrics



with open(f"{output_path}/testsuite_results.json", 'w') as f:
    json.dump(metrics, f, indent=4)





