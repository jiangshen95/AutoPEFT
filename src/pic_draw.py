import json
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

json_path = 'results/lora_low_lr.json'
save_path = 'results/lora32_full'

def plot_aggregated_metrics(file_path, output_dir='pic'):
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    strategy_metrics = defaultdict(lambda: {'eval_loss': [], 'eval_accuracy': []})
    dataset_names = []

    for i in range(0, len(lines), 2):
        dataset_name = lines[i].strip()
        dataset_names.append(dataset_name)
        data = json.loads(lines[i + 1].strip())

        for strategy, metrics in data.items():
            strategy_metrics[strategy]['eval_loss'].append(metrics['eval_loss'])
            strategy_metrics[strategy]['eval_accuracy'].append(metrics['eval_accuracy'])

    strategies = list(strategy_metrics.keys())
    mean_losses = [np.mean(strategy_metrics[strategy]['eval_loss']) for strategy in strategies]
    mean_accuracies = [np.mean(strategy_metrics[strategy]['eval_accuracy']) for strategy in strategies]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Mean Eval Loss', color=color)
    bars = ax1.bar(strategies, mean_losses, color=color, alpha=0.6, label='Mean Eval Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(strategies, rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Eval Accuracy', color=color)
    line = ax2.plot(strategies, mean_accuracies, color=color, marker='o', linestyle='-', label='Mean Eval Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    for i, v in enumerate(mean_accuracies):
        ax2.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    plt.title('Mean Eval Loss and Accuracy across Datasets')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.savefig(os.path.join(output_dir, 'aggregated_metrics.png'))
    plt.close()

# 示例调用
plot_aggregated_metrics(json_path, save_path)

def plot_metrics(file_path, output_dir='pic'):
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        dataset_name = lines[i].strip()
        data = json.loads(lines[i + 1].strip())

        strategies = list(data.keys())
        eval_losses = [data[strategy]['eval_loss'] for strategy in strategies]
        eval_accuracies = [data[strategy]['eval_accuracy'] for strategy in strategies]

        mean_loss = np.mean(eval_losses)
        mean_accuracy = np.mean(eval_accuracies)

        strategies.append('Mean')
        eval_losses.append(mean_loss)
        eval_accuracies.append(mean_accuracy)

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Eval Loss', color=color)
        bars = ax1.bar(strategies, eval_losses, color=color, alpha=0.6, label='Eval Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticklabels(strategies, rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Eval Accuracy', color=color)
        line = ax2.plot(strategies, eval_accuracies, color=color, marker='o', linestyle='-', label='Eval Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        for i, v in enumerate(eval_accuracies):
            ax2.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        fig.tight_layout()
        plt.title(f'{dataset_name} - Eval Loss and Accuracy')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_metrics.png'))
        plt.close()

# 示例调用
plot_metrics(json_path, save_path)