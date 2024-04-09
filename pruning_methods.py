import torch

# 统计并排序模型中每层LoRA矩阵中0的个数
def count_and_sort_zeros_in_qkv_lora(model):
    layer_zero_counts = {}  # 初始化一个字典来存储每层的0计数
    for name, param in model.named_parameters():  # 遍历模型的所有参数
        # 仅处理包含LoRA参数的层（query, key, value）
        if "loras" in name and ("query" in name or "key" in name or "value" in name):
            layer_num = name.split('.')[3]  # 提取层编号
            zero_count = (param == 0).sum().item()  # 计算当前参数中0的个数
            # 累加当前层的0计数
            if layer_num in layer_zero_counts:
                layer_zero_counts[layer_num] += zero_count
            else:
                layer_zero_counts[layer_num] = zero_count
    # 根据0的个数对层进行排序
    sorted_zero_counts = sorted(layer_zero_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_zero_counts

# 统计并排序模型中每层LoRA矩阵中小于给定阈值的值的个数
def count_and_sort_values_below_threshold_in_qkv_lora(model, threshold):
    layer_value_counts = {}  # 初始化一个字典来存储每层的小于阈值的值的计数
    for name, param in model.named_parameters():  # 遍历模型的所有参数
        # 仅处理包含LoRA参数的层（query, key, value）
        if "loras" in name and ("query" in name or "key" in name or "value" in name):
            layer_num = name.split('.')[3]  # 提取层编号
            # 计算绝对值小于阈值的元素个数
            value_count = (param.abs() < threshold).sum().item()
            # 累加当前层的小于阈值的值的计数
            if layer_num in layer_value_counts:
                layer_value_counts[layer_num] += value_count
            else:
                layer_value_counts[layer_num] = value_count
    # 根据小于阈值的值的个数对层进行排序
    sorted_value_counts = sorted(layer_value_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_value_counts

def calculate_and_sort_minimum_weight_in_qkv_lora(model):
    layer_minimum_weights = {}  # 初始化一个字典来存储每层的最小权重
    for name, param in model.named_parameters():  # 遍历模型的所有参数
        # 仅处理包含LoRA参数的层（query, key, value）
        if "loras" in name and ("query" in name or "key" in name or "value" in name):
            layer_num = name.split('.')[3]  # 提取层编号
            # 计算所有元素的平方和再除以元素个数
            minimum_weight = (param.pow(2).sum() / param.numel()).item()
            # 存储当前层的最小权重
            layer_minimum_weights[layer_num] = minimum_weight
    # 根据最小权重对层进行排序
    sorted_minimum_weights = sorted(layer_minimum_weights.items(), key=lambda x: x[1], reverse=True)
    return sorted_minimum_weights

# 将0数量最多的层中的Q、K、V矩阵中的LoRA置0，并且将该层设为不可训练
def zero_out_top_lora_layer(model, sorted_zero_counts):
    if not sorted_zero_counts:  # 如果没有层需要处理，则直接返回
        print("No layers to process.")
        return
    top_layer_num = sorted_zero_counts[0][0]  # 获取0数量最多的层的编号
    for name, param in model.named_parameters():  # 遍历模型的所有参数
        # 仅处理目标层的LoRA参数（query, key, value）
        if f"layer.{top_layer_num}" in name and "loras" in name and ("query" in name or "key" in name or "value" in name):
            with torch.no_grad():  # 修改参数时不计算梯度
                param.zero_()  # 将参数全部置为0
            param.requires_grad = False  # 设置为不可训练