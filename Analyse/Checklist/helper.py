from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from checklist.pred_wrapper import PredictorWrapper
import torch
import itertools
import json
import pandas as pd
from tabulate import tabulate
from IPython.display import display, Markdown
import uuid
import os
import checklist
from checklist.test_suite import TestSuite
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
import random


def crossproduct(t):
    # takes the output of editor.template and does the cross product of contexts and qas
    ret = []
    ret_labels = []
    for x in t.data:
        cs = x['contexts']
        qas = x['qas']
        d = list(itertools.product(cs, qas))
        ret.append([(x[0], x[1][0]) for x in d])
        ret_labels.append([x[1][1] for x in d])
    t.data = ret
    t.labels = ret_labels
    return t

def format_squad_with_context(x, pred, conf, label=None, *args, **kwargs):
    c, q = x
    ret = 'C: %s\nQ: %s\n' % (c, q)
    if label is not None:
        ret += 'A: %s\n' % label
    ret += 'P: %s\n' % pred
    return ret

def get_finetuned_electra_predictor(model_path="../../Data/trained_model"):
    """
    Returns a wrapped predictor compatible with CheckList for your finetuned ELECTRA QA model.

    Input: list of (context, question) tuples
    Output: list of predicted answer strings (exact original text)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()  # set to evaluation mode

    def predict_fn(examples):
        """
        examples: list of (context, question) tuples
        returns: list of answer strings
        """
        answers = []
        for context, question in examples:
            # Encode inputs and get offsets for exact character mapping
            inputs = tokenizer(
                question, context,
                return_tensors="pt",
                truncation=True,
                return_offsets_mapping=True
            )
            offset_mapping = inputs.pop("offset_mapping")[0]  # shape: [seq_len, 2]

            with torch.no_grad():
                outputs = model(**inputs)

            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits)

            # Use offsets to extract exact substring from context
            start_char = offset_mapping[start_idx][0].item()
            end_char = offset_mapping[end_idx][1].item()
            answer = context[start_char:end_char]

            if answer.strip() == "":
                answer = "empty"

            answers.append(answer)

        return answers

    # Wrap predictor for CheckList
    wrapped_predictor = PredictorWrapper.wrap_predict(predict_fn)
    return wrapped_predictor

def show_example(test_cases, n=1):
    model = get_finetuned_electra_predictor()
    for i in range(n):
        pred = model(test_cases.data[i])[0]
        labels = test_cases.labels[i]
        for j, item in enumerate(test_cases.data[i]):
            print(f"  [{j+1}] {item} , pred: {pred[j]} , label: {labels[j]}")
        print('---')


def export_suite_to_jsonl(suite, output_file="checklist_testsuite.jsonl"):
    all_rows = []
    used_ids = set()  # Track used IDs to ensure uniqueness

    for testname in suite.tests:                  
        test = suite.tests[testname]
        all_predictions = test.results.preds
        all_labels = test.labels

        for testcase_idx, (testcase, pred_array, label_array) in enumerate(zip(test.data, all_predictions, all_labels)):

            for qa_idx, ((context, question), pred, label) in enumerate(zip(testcase, pred_array, label_array)):
                
                # Generate unique ID with collision check
                while True:
                    new_id = uuid.uuid4().hex[:24]
                    if new_id not in used_ids:
                        used_ids.add(new_id)
                        break
            
                all_rows.append({
                    'id': new_id,
                    'title': testname,
                    'context': context,
                    'question': question,
                    'answers':{
                        'text':[label],
                        'answer_start':[context.find(label)]
                    },
                    'pred': pred,
                    'test_details':{
                        'test_type': test.__class__.__name__,
                        'capability': test.capability,
                        'description': test.description,
                        'name': test.name,
                        'testcase_id': testcase_idx, 
                        'qa_pair_index': qa_idx,                     # Which QA pair in this testcase
                        'total_qa_pairs': len(testcase),             # Total QA pairs in this testcase
                        'expect_function': test.expect.__name__ if test.expect else None
                    }
                })

        

    with open(output_file, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Saved, {len(all_rows)} rows to {output_file}.")
    print(f"🔒 Generated {len(used_ids)} unique IDs (collision-free)")


def generate_dataset(suite, output_folder="ChecklistData", train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Generate train/test/validation splits from a test suite ensuring:
    1. No data leakage (examples only appear in one split)
    2. QA pairs from same testcase stay together
    3. Equal splits across different test names (70% train, 20% test, 10% validation for each test)
    4. Proper rounding to maintain constraints
    
    Args:
        suite: CheckList test suite
        output_folder: folder path to save the splits
        train_ratio: ratio for training split (default 0.7)
        test_ratio: ratio for test split (default 0.2)
        val_ratio: ratio for validation split (default 0.1)
    """
    
    # Ensure ratios sum to 1.0
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio}")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    # Step 1: Export suite to temporary combined file using existing function
    temp_combined_file = os.path.join(output_folder, "temp_combined.jsonl")
    export_suite_to_jsonl(suite, temp_combined_file)
    
    # Step 2: Read the exported file and group by test name and testcase
    testcases_by_test = {}
    
    with open(temp_combined_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            test_name = item['title']
            testcase_id = item['test_details']['testcase_id']
            
            if test_name not in testcases_by_test:
                testcases_by_test[test_name] = {}
            
            if testcase_id not in testcases_by_test[test_name]:
                testcases_by_test[test_name][testcase_id] = []
            
            testcases_by_test[test_name][testcase_id].append(item)
    
    # Step 3: Split each test name separately to ensure equal splits
    train_data = []
    test_data = []
    val_data = []
    
    print(f"📊 Splitting each test separately with {train_ratio:.1%}/{test_ratio:.1%}/{val_ratio:.1%} train/test/val ratios:")
    
    for test_name, testcases in testcases_by_test.items():
        # Group QA pairs by testcase (keep pairs together)
        testcase_groups = []
        for testcase_id in sorted(testcases.keys()):
            qa_pairs = testcases[testcase_id]
            qa_pairs.sort(key=lambda x: x['test_details']['qa_pair_index'])
            testcase_groups.append(qa_pairs)
        
        # Shuffle testcase groups for this test
        random.shuffle(testcase_groups)
        
        # Calculate split for this test
        total_groups = len(testcase_groups)
        train_groups_count = round(total_groups * train_ratio)
        test_groups_count = round(total_groups * test_ratio)
        val_groups_count = total_groups - train_groups_count - test_groups_count  # Remaining for validation
        
        # Split the testcase groups for this test
        test_train_groups = testcase_groups[:train_groups_count]
        test_test_groups = testcase_groups[train_groups_count:train_groups_count + test_groups_count]
        test_val_groups = testcase_groups[train_groups_count + test_groups_count:]
        
        # Add to overall splits
        for group in test_train_groups:
            train_data.extend(group)
        
        for group in test_test_groups:
            test_data.extend(group)
        
        for group in test_val_groups:
            val_data.extend(group)
        
        # Print per-test statistics
        train_examples = sum(len(group) for group in test_train_groups)
        test_examples = sum(len(group) for group in test_test_groups)
        val_examples = sum(len(group) for group in test_val_groups)
        total_examples = train_examples + test_examples + val_examples
        
        if total_examples > 0:
            actual_train_ratio = train_examples / total_examples
            actual_test_ratio = test_examples / total_examples
            actual_val_ratio = val_examples / total_examples
        else:
            actual_train_ratio = actual_test_ratio = actual_val_ratio = 0
        
        print(f"   {test_name}: {len(test_train_groups)}/{len(test_test_groups)}/{len(test_val_groups)} groups "
              f"({actual_train_ratio:.1%}/{actual_test_ratio:.1%}/{actual_val_ratio:.1%} examples)")
    
    # Step 4: Save splits to files
    train_file = os.path.join(output_folder, "checklist_train.jsonl")
    test_file = os.path.join(output_folder, "checklist_test.jsonl")
    val_file = os.path.join(output_folder, "checklist_validation.jsonl")
    
    with open(train_file, "w") as f:
        for row in train_data:
            f.write(json.dumps(row) + "\n")
    
    with open(test_file, "w") as f:
        for row in test_data:
            f.write(json.dumps(row) + "\n")
    
    with open(val_file, "w") as f:
        for row in val_data:
            f.write(json.dumps(row) + "\n")
    
    # Step 5: Delete the temporary combined file
    os.remove(temp_combined_file)
    
    # Step 6: Print overall summary
    total_examples = len(train_data) + len(test_data) + len(val_data)
    if total_examples > 0:
        actual_train_ratio = len(train_data) / total_examples
        actual_test_ratio = len(test_data) / total_examples
        actual_val_ratio = len(val_data) / total_examples
    else:
        actual_train_ratio = actual_test_ratio = actual_val_ratio = 0
    
    total_tests = len(testcases_by_test)
    
    print(f"\n📋 Overall Dataset Split Summary:")
    print(f"   Total test names: {total_tests}")
    print(f"   Total examples: {total_examples}")
    print(f"   Train examples: {len(train_data)} ({actual_train_ratio:.1%})")
    print(f"   Test examples: {len(test_data)} ({actual_test_ratio:.1%})")
    print(f"   Validation examples: {len(val_data)} ({actual_val_ratio:.1%})")
    print(f"✅ Saved train split to: {train_file}")
    print(f"✅ Saved test split to: {test_file}")
    print(f"✅ Saved validation split to: {val_file}")
    print(f"🗑️  Temporary combined file deleted")
    
    return train_data, test_data, val_data
   
def get_summary(suite):
    summary_list = []

    for testname, test in suite.tests.items():
        stats = test.get_stats()
        n_examples = sum(len(group) for group in test.data)
        
        summary_list.append({
            "Test Name": testname,
            "Total Cases": stats.testcases,
            "Example Per Case": f"{n_examples/stats.testcases:.2f}",
            "Failures": stats.fails,
            "Failure Rate": f"{stats.fail_rate:.2f}%",
        })

    return pd.DataFrame(summary_list)

def get_detailed_summary(suite):
    """
    Get detailed summary with average failure rate per example instead of per testcase.
    
    For each test:
    - Calculate failure rate per example (how many QA pairs failed in each testcase)
    - Average these failure rates across all testcases
    
    Returns DataFrame with additional 'Average Failure' column
    """
    summary_list = []

    for testname, test in suite.tests.items():
        stats = test.get_stats()
        n_examples = sum(len(group) for group in test.data)
        
        # Calculate average failure rate per example
        average_failure = 0.0
        if hasattr(test, 'results') and test.results and hasattr(test.results, 'preds'):
            total_testcases = len(test.data)
            if total_testcases > 0:
                testcase_failure_rates = []
                
                for testcase_idx, (testcase_data, testcase_preds, testcase_labels) in enumerate(
                    zip(test.data, test.results.preds, test.labels)
                ):
                    # Count failures in this testcase
                    failures_in_testcase = 0
                    total_qa_pairs = len(testcase_data)
                    
                    for qa_idx, (pred, label) in enumerate(zip(testcase_preds, testcase_labels)):
                        if hasattr(test, 'expect') and test.expect:
                            # Use the test's expect function if available
                            # CheckList expect functions typically take (x, pred, conf, label=None, meta=None)
                            try:
                                if not test.expect(testcase_data[qa_idx], pred, None, label=label, meta=None):
                                    failures_in_testcase += 1
                            except TypeError:
                                # Fallback: try with just the prediction result
                                try:
                                    if not test.expect(pred):
                                        failures_in_testcase += 1
                                except:
                                    # Final fallback: simple comparison
                                    if pred != label:
                                        failures_in_testcase += 1
                        else:
                            # Default comparison
                            if pred != label:
                                failures_in_testcase += 1
                    
                    # Calculate failure rate for this testcase (0.0 to 1.0)
                    testcase_failure_rate = failures_in_testcase / total_qa_pairs if total_qa_pairs > 0 else 0.0
                    testcase_failure_rates.append(testcase_failure_rate)
                
                # Average failure rate across all testcases
                average_failure = sum(testcase_failure_rates) / len(testcase_failure_rates) if testcase_failure_rates else 0.0
        
        summary_list.append({
            "Test Name": testname,
            "Total Cases": stats.testcases,
            "Example Per Case": f"{n_examples/stats.testcases:.2f}",
            "Failures": stats.fails,
            "Failure Rate": f"{stats.fail_rate:.2f}%",
            "Average Failure": f"{average_failure:.2%}",
        })

    return pd.DataFrame(summary_list)


def display_and_export_mdtable(dataframe, do_display= True, do_export=True, output_file="checklist_testsuite_summary.md"):
    md_table = tabulate(dataframe, headers='keys', tablefmt='github', showindex=False)
    if do_display:
        display(Markdown(md_table))

    if do_export:
        with open(output_file, "w") as f:
            f.write(md_table)


def clean(string):
    return string.lstrip('[a,the,an,in,at] ').rstrip('.')
def expect_squad(x, pred, conf, label=None, meta=None):
    return clean(pred) == clean(label)

class ChecklistTestRunner:
    """
    Class to handle testing with .jsonl datasets using Checklist
    """
    
    def __init__(self, model_path="../../Data/trained_model"):
        self.model_predictor = get_finetuned_electra_predictor(model_path)
        
    def load_jsonl_by_test_groups(self, jsonl_file):
        """
        Load .jsonl file and group by testcase_id to recreate original structure
        Returns dict of {test_name: {'data': [...], 'labels': [...], 'test_config': {...}}}
        """
        # Group by test name and testcase_id directly
        testcases_by_group = {}
        
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    test_name = item['title']
                    testcase_id = item['test_details']['testcase_id']
                    
                    if test_name not in testcases_by_group:
                        testcases_by_group[test_name] = {
                            'testcases': {},
                            'test_config': item['test_details']
                        }
                    
                    if testcase_id not in testcases_by_group[test_name]['testcases']:
                        testcases_by_group[test_name]['testcases'][testcase_id] = []
                    
                    testcases_by_group[test_name]['testcases'][testcase_id].append(item)
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing line {line_num}: {e}")
                    continue
        
        # Convert to final format and discard intermediate structure
        test_groups = {}
        for test_name, group_data in testcases_by_group.items():
            data = []
            labels = []
            
            for testcase_id in sorted(group_data['testcases'].keys()):
                qa_pairs = group_data['testcases'][testcase_id]
                qa_pairs.sort(key=lambda x: x['test_details']['qa_pair_index'])
                
                testcase_data = [(item['context'], item['question']) for item in qa_pairs]
                testcase_labels = [item['answers']['text'][0] for item in qa_pairs]
                
                data.append(testcase_data)
                labels.append(testcase_labels)
            
            # Only store what's actually used
            test_groups[test_name] = {
                'data': data,
                'labels': labels, 
                'test_config': group_data['test_config']
            }
        
        return test_groups
    
    def create_suite_from_jsonl(self, jsonl_file):
        """
        Create a complete test suite from .jsonl file using original test configurations
        """
        suite = TestSuite()
        test_groups = self.load_jsonl_by_test_groups(jsonl_file)
        
        print(f"📊 Found {len(test_groups)} test groups in {jsonl_file}")
        
        # Create expectation function (avoid naming conflict)
        expect_squad_fn = Expect.single(expect_squad)
        
        for test_name, test_data in test_groups.items():
            config = test_data['test_config']
            data = test_data['data']
            labels = test_data['labels']
            
            # Create test dynamically based on stored config
            test_class = {'MFT': MFT, 'INV': INV, 'DIR': DIR}[config['test_type']]
            
            test = test_class(
                data=data,
                labels=labels,
                name=config['name'],
                capability=config['capability'],
                description=config['description']
            )
            
            # Apply expect function if needed
            if config.get('expect_function') in ['expect_squad', 'expect']:
                test.expect = expect_squad_fn
            
            print(f"  - {test_name}: {len(data)} testcases, {sum(len(tc) for tc in data)} examples")
            suite.add(test)
        
        return suite
    
    def run_suite_on_jsonl(self, jsonl_file, save_results=True, output_folder="results"):
        """
        Complete pipeline: load .jsonl -> create suite -> run tests -> save results
        """
        print(f"🔄 Processing {jsonl_file}...")
        
        # Create suite with original test configurations
        suite = self.create_suite_from_jsonl(jsonl_file)
        
        # Run tests
        print("🧪 Running tests with model predictions...")
        suite.run(self.model_predictor, overwrite=True)
        
        # Display summary
        print("\n📋 Test Results Summary:")
        suite.summary(n=1)

        summary_df = get_detailed_summary(suite)
        # Get detailed summary
        if save_results:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        
            display_and_export_mdtable(
                summary_df, 
                do_display=True, 
                do_export=True,
                output_file=os.path.join(output_folder, "summary.md")
            )
            
            print(f"💾 Saving results to {output_folder}/...")
            suite.save(os.path.join(output_folder, "test_suite.pkl"))
            export_suite_to_jsonl(suite, os.path.join(output_folder, "detailed_results.jsonl"))
            print(f"✅ All results saved in folder: {output_folder}")
        
        return summary_df
    
