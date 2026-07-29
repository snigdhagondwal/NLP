import argparse
from helper import load_and_process_predictions
from ques_type import get_q_type, classify_decomposition_type, detect_multi_questions


parser = argparse.ArgumentParser(description="input predictions file and category of questions")
parser.add_argument("--predictions_file", type=str, help="predictions file path", required=True)
parser.add_argument("--category_func", choices=['question_type','compositional_type', 'multihop'],default=None, help="predictions category function")

args = parser.parse_args()
print(f"Prediction file: {args.predictions_file}")
print(f"Category: {args.category_func}")

category_funtions = {'question_type': get_q_type,
                    'compositional_type': classify_decomposition_type,
                    'multihop': detect_multi_questions,
                    None: None
}    
category_function = category_funtions[args.category_func]


summary = load_and_process_predictions(args.predictions_file, category_function)
print(summary)