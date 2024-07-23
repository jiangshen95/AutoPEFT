import os
import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = '/home/autopeft/AutoPEFT/outputs/2024-07-22/output_2024-07-22_16:11:01.log'

# 结果存储字典
results = {}
dataset_counter = 0  # 数据集计数器

# 读取日志文件
with open(log_file_path, 'r') as file:
    lines = file.readlines()

current_dataset = None
current_prune_method = None

# 解析日志文件
for line in lines:
    if 'Final-Result' in line:
        match = re.search(r"Final-Result.*?({'eval_loss':.*?})", line)
        if match:
            result = eval(match.group(1))
            current_dataset = dataset_counter
            dataset_counter += 1
            if current_dataset not in results:
                results[current_dataset] = {}
            current_prune_method = None
    elif 'Prune method' in line:
        match = re.search(r"Prune method (\w+)", line)
        if match:
            current_prune_method = match.group(1)
            if current_prune_method not in results[current_dataset]:
                results[current_dataset][current_prune_method] = {'eval_loss': [], 'eval_accuracy': []}
    elif 'Val-Result' in line:
        match = re.search(r"Val-Result.*?({'eval_loss':.*?})", line)
        if match and current_prune_method is not None:
            result = eval(match.group(1))
            results[current_dataset][current_prune_method]['eval_loss'].append(result['eval_loss'])
            results[current_dataset][current_prune_method]['eval_accuracy'].append(result['eval_accuracy'])


# 创建保存图表的文件夹
output_dir = '/home/autopeft/AutoPEFT/results/search_turn_lora'
os.makedirs(output_dir, exist_ok=True)

# 绘制和保存图表
for dataset, prune_methods in results.items():
    for prune_method, metrics in prune_methods.items():
        epochs = range(1, len(metrics['eval_loss']) + 1)

        max_loss = min(metrics['eval_loss'])
        max_accuracy = max(metrics['eval_accuracy'])
        max_loss_index = metrics['eval_loss'].index(max_loss)
        max_accuracy_index = metrics['eval_accuracy'].index(max_accuracy)
        
        plt.figure()
        plt.plot(epochs, metrics['eval_loss'], label='Loss', marker='o')
        plt.plot(epochs, metrics['eval_accuracy'], label='Accuracy', marker='o')
        
        # 在图表上标注具体数值，并凸显最大值的点
        for i, (loss, accuracy) in enumerate(zip(metrics['eval_loss'], metrics['eval_accuracy'])):
            if i == max_loss_index:
                plt.annotate(f'{loss:.2f}', (epochs[i], loss), textcoords="offset points", xytext=(0,5), ha='center', fontsize=12, fontweight='bold', color='red')
                plt.scatter(epochs[i], loss, color='red', s=100, zorder=5)
            else:
                plt.annotate(f'{loss:.2f}', (epochs[i], loss), textcoords="offset points", xytext=(0,5), ha='center')
            
            if i == max_accuracy_index:
                plt.annotate(f'{accuracy:.2f}', (epochs[i], accuracy), textcoords="offset points", xytext=(0,5), ha='center', fontsize=12, fontweight='bold', color='blue')
                plt.scatter(epochs[i], accuracy, color='blue', s=100, zorder=5)
            else:
                plt.annotate(f'{accuracy:.2f}', (epochs[i], accuracy), textcoords="offset points", xytext=(0,5), ha='center')
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f'Dataset: {dataset}\nPrune Method: {prune_method}')
        plt.legend()
        
        # 保存图表
        output_path = os.path.join(output_dir, f'dataset_{dataset}_prune_{prune_method}.png')
        plt.savefig(output_path)
        plt.close()
        
# 绘制和保存每个剪枝方法在所有数据集上的折线图
prune_methods_all = {}

for dataset, prune_methods in results.items():
    for prune_method, metrics in prune_methods.items():
        if prune_method not in prune_methods_all:
            prune_methods_all[prune_method] = {'eval_loss': [], 'eval_accuracy': [], 'epochs': []}
        prune_methods_all[prune_method]['eval_loss'].append(metrics['eval_loss'])
        prune_methods_all[prune_method]['eval_accuracy'].append(metrics['eval_accuracy'])
        prune_methods_all[prune_method]['epochs'].append(range(1, len(metrics['eval_loss']) + 1))

for prune_method, data in prune_methods_all.items():
    plt.figure(figsize=(10, 6))
    for i, (epochs, loss, accuracy) in enumerate(zip(data['epochs'], data['eval_loss'], data['eval_accuracy'])):
        plt.plot(epochs, accuracy, label=f'Dataset {i+1} Accuracy', marker='o')
        
        # 在图表上标注具体数值，并凸显最大值的点
        max_accuracy = max(accuracy)
        max_accuracy_index = accuracy.index(max_accuracy)
        
        for j, (l, a) in enumerate(zip(loss, accuracy)):
            if j == max_accuracy_index:
                plt.annotate(f'{a:.2f}', (epochs[j], a), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10, fontweight='bold', color='blue')
                plt.scatter(epochs[j], a, color='blue', s=100, zorder=5)
            else:
                plt.annotate(f'{a:.2f}', (epochs[j], a), textcoords="offset points", xytext=(0,5), ha='center')
        
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Prune Method: {prune_method} Across All Datasets')
    plt.legend()
    
    # 保存图表
    output_path = os.path.join(output_dir, f'prune_{prune_method}_all_datasets.png')
    plt.savefig(output_path)
    plt.close()