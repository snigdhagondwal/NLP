
import json
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def read_models_data():
    '''
    Reads the testsuite_results.json files from all model result directories
    located in ../Results/ and returns a dictionary with model names as keys
    and their corresponding data as values.
    '''
    models_data = {}
    for model_path in Path("../Results").iterdir():
        if model_path.is_dir():
            json_file = model_path / "testsuite_results.json"
            if json_file.exists():
                with open(json_file) as f:
                    models_data[model_path.name] = json.load(f)
    return models_data

def plot_custom_metrics(models_data, ax, dataset_name='squad'):
    '''
    This mainly plots bar charts for F1 and Exact Match scores for different models on a given dataset.
    need to take ax as argument to plot on specific subplot
    '''
    squad_data = []
    for model, data in models_data.items():
        if dataset_name in data:
            squad_data.append({
                'Model': model,
                'F1': data[dataset_name]['eval_f1'],
                'Exact Match': data[dataset_name]['eval_exact_match']
            })
    
    df = pd.DataFrame(squad_data)
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['F1'], width, label='F1 Score', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['Exact Match'], width, label='Exact Match', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name.title()} Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_adversarial_metrics(models_data, ax, dataset_name='squad', type='correct'):
    '''
    plots bar charts for adversarial analysis metrics (question_type, compositional_type, multihop) for different models on a given dataset.
    need to take ax as argument to plot on specific subplot
    also need to specify type of ratio to plot: 'correct', 'failed', 'partially_correct'
    and dataset_name: 'squad' or 'adversarial'
    '''

    adv_data = []
    for model, data in models_data.items():
        if dataset_name in data:
            # Extract the 'type' values from each category (e.g., 'correct', 'failed', 'partially_correct')
            question_type = data[dataset_name]['question_type'][type]
            compositional_type = data[dataset_name]['compositional_type'][type]
            multihop = data[dataset_name]['multihop'][type]
            
            adv_data.append({
                'Model': model,
                'Question Type': question_type,
                'Compositional Type': compositional_type,
                'Multihop': multihop
            })
    
    df = pd.DataFrame(adv_data)
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['Question Type'], width, label='Question Type', alpha=0.8)
    bars2 = ax.bar(x, df['Compositional Type'], width, label='Compositional Type', alpha=0.8)
    bars3 = ax.bar(x + width, df['Multihop'], width, label='Multihop', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars3:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel(f'{type} Ratio')
    ax.set_title(f'{dataset_name.title()} Adversarial Analysis - {type} Ratios')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_checklist_heatmap(models_data, ax, metric='failure_rate'):
    # Create a matrix: rows = test names, columns = models
    checklist_data = []
    
    for model, data in models_data.items():
        if 'checklist' in data:
            for test_name, metrics in data['checklist'].items():
                # Convert percentage string to float
                value = float(metrics[metric].rstrip('%'))
                checklist_data.append({
                    'Model': model,
                    'Test': test_name,
                    'Value': value
                })
    
    df = pd.DataFrame(checklist_data)
    pivot_df = df.pivot(index='Test', columns='Model', values='Value')
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, cmap='RdYlBu_r', ax=ax, 
                fmt='.1f', cbar_kws={'label': f'{metric} (%)'})
    ax.set_title(f'Checklist Tests - {metric.replace("_", " ").title()}')
    ax.set_xlabel('Models')
    ax.set_ylabel('Test Categories')



def create_custom_dashboard(models_data):
    '''
    Creates a dashboard with subplots for:
    1. Squad Performance (F1 and Exact Match)
    2. Adversarial Performance (F1 and Exact Match)
    3. Adversarial Analysis on Squad Dataset (Correct Ratios)
    4. Adversarial Analysis on Adversarial Dataset (Correct Ratios)
    '''
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    plot_custom_metrics(models_data, axes[0,0],'squad')
    plot_custom_metrics(models_data, axes[0,1],'adversarial')
    plot_adversarial_metrics(models_data, axes[1,0],'squad', type='correct')
    plot_adversarial_metrics(models_data, axes[1,1],'adversarial', type='correct')
    plt.tight_layout()
    # plt.savefig("model_comparison_dashboard.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_checklist_dashboard(models_data):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    plot_checklist_heatmap(models_data, axes[0], 'failure_rate')
    plot_checklist_heatmap(models_data, axes[1], 'average_failure')
    
    plt.tight_layout()
    plt.show()