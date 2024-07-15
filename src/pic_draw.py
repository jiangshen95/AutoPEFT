import json
import matplotlib.pyplot as plt
import os


def plot_metrics(file_path, output_dir='pic'):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 跳过第一行时间信息
    # lines = lines[1:]

    for i in range(0, len(lines), 2):
        dataset_name = lines[i].strip()
        data = json.loads(lines[i + 1].strip())

        strategies = list(data.keys())
        eval_losses = [data[strategy]['eval_loss'] for strategy in strategies]
        eval_accuracies = [
            data[strategy]['eval_accuracy'] for strategy in strategies
        ]

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Eval Loss', color=color)
        bars = ax1.bar(
            strategies, eval_losses, color=color, alpha=0.6, label='Eval Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticklabels(strategies, rotation=45)

        # 在柱状图上标出eval_loss的值
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom')

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Eval Accuracy', color=color)
        line = ax2.plot(
            strategies,
            eval_accuracies,
            color=color,
            marker='o',
            linestyle='-',
            label='Eval Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        # 在折线图上标出eval_accuracy的值
        for i, v in enumerate(eval_accuracies):
            ax2.annotate(
                f'{v:.4f}',
                xy=(i, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom')

        fig.tight_layout()
        plt.title(f'{dataset_name} - Eval Loss and Accuracy')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_metrics.png'))
        plt.close()


# 示例调用
plot_metrics('/home/autopeft/AutoPEFT/results/final-lora-adapter.json',
             '/home/autopeft/AutoPEFT/results/pic-adapter-LoRA')
