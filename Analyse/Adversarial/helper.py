import os
import json
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from .ques_type import get_q_type, classify_decomposition_type, detect_multi_questions
nltk.download('punkt')

def normalize(text):
    if text is None:
        return ""
    return " ".join(text.lower().strip().split())


def exact_match(pred, gold_list):
    pred_n = normalize(pred)
    return int(any(pred_n == normalize(g) for g in gold_list))


def f1_score(pred, gold):
    pred_tokens = word_tokenize(pred.lower())
    gold_tokens = word_tokenize(gold.lower())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def max_f1(pred, gold_list):
    return max(f1_score(pred, g) for g in gold_list)

def categorize(em, f1):
    if em == 1:
        return "correct"
    elif f1 >= 0.5:
        return "partially correct"
    else:
        return "failed"


def load_and_process_predictions(file_path, category_type):
    rows = []
    with open(file_path) as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    if category_type!= None:
        df["multi_question"] = df["question"].apply(category_type)
    
    df["exact_match"] = df.apply(lambda r: exact_match(r["predicted_answer"], r["answers"]["text"]), axis=1)
    df["f1"] = df.apply(lambda r: max_f1(r["predicted_answer"], r["answers"]["text"]), axis=1)
    df["result_type"] = df.apply(lambda r: categorize(r["exact_match"], r["f1"]), axis=1)

    if category_type!= None:
        summary = df.groupby(["multi_question", "result_type"]).size().unstack(fill_value=0)
    else:
        summary = df.groupby(["result_type"]).size()#.unstack(fill_value=0)
    # print(summary)
    return summary


def save_markdown(summaries, category_names, title, output_file):
    """
    Save multiple summary data as markdown format with title header, category headers and tables.
    
    Args:
        summaries: list of pandas DataFrame or Series returned by load_and_process_predictions
        category_names: list of category names for the headers (must match length of summaries)
        title: main title for the entire section
        output_file: path to the output markdown file (will be appended to by default)
    """
    
    if len(summaries) != len(category_names):
        raise ValueError("Number of summaries must match number of category names")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a') as f:
        # Write main title header
        f.write(f"# {title}\n\n")
        
        for i, (summary, category_name) in enumerate(zip(summaries, category_names)):
            # Write category header
            f.write(f"## {category_name}\n\n")
            
            # Convert summary to markdown table
            if isinstance(summary, pd.DataFrame):
                # For DataFrame (when category function is provided)
                markdown_table = summary.to_markdown()
            elif isinstance(summary, pd.Series):
                # For Series (when no category function is provided)
                # Convert to DataFrame for better table formatting
                df_summary = summary.to_frame(name='Count')
                markdown_table = df_summary.to_markdown()
            else:
                raise ValueError("Summary must be a pandas DataFrame or Series")
            
            f.write(markdown_table)
            f.write("\n\n")
            
           
        f.write("---\n\n")


def macro_anallysis(df):
    df_ratios = df.apply(lambda x: x / x.sum() if x.sum() != 0 else x * 0, axis=1)
    column_averages = df_ratios.mean()
    return column_averages.to_dict()


def run_analysis(predictions_file, output_file='Adversarial_Test_Suite_Report.md', model_name=None):
    category_funtions = {'question_type': get_q_type,
                    'compositional_type': classify_decomposition_type,
                    'multihop': detect_multi_questions,
                    None: None}    

    summaries=[]
    macro_analysis_metrics={}
    for category in category_funtions.keys():
            category_function = category_funtions[category]
            summary = load_and_process_predictions(predictions_file, category_function)
            if category:
                macro_analysis_metrics[category] = macro_anallysis(summary)
            summaries.append(summary)

    title = f"Adversarial Test Suite Analysis{' - ' + model_name if model_name else ''}"
    save_markdown(summaries, list(category_funtions.keys()), title, output_file)
    return macro_analysis_metrics