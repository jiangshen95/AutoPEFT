import torch
import re
from torch.nn.parameter import Parameter
import logging

logger = logging.getLogger('lab')


def get_trainable_parameters(model):
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # logger.info(name)
            names.append(name)
    return names


def group_parameters_by_prefix(names, opts=[], print_names=False, task_name=''):
    groups = {}
    # 修改条件，确保name包含opts列表中的任一元素
    names = [
        name for name in names
        if task_name in name and 'head' not in name and any(
            opt in name for opt in opts)
    ]
    for name in names:
        # 分割参数名，获取前缀
        prefix = name.split(task_name)[0]
        prefix = prefix.replace('query.', '').replace('value.', '')
        # 如果前缀已经在字典中，就将参数名添加到对应的列表中
        if prefix in groups:
            groups[prefix].append(name)
        # 否则，创建一个新的列表
        else:
            groups[prefix] = [name]
    if print_names:
        for prefix, names in groups.items():
            logger.info(f"{prefix}:")
            for name in names:
                logger.info(f"  {name}")
    return groups


def find_group_with_most_small_values(groups, model, p_method, top_p=1):
    group_values = []
    if p_method == 'zeros':
        for group, names in groups.items():
            num_zeros = sum(
                (model.state_dict()[name] == 0).sum().item() for name in names)
            group_values.append((group, names, num_zeros))
    elif p_method == 'values_below_threshold':
        threshold = 0.001  # 可以根据需要调整阈值
        for group, names in groups.items():
            num_values_below_threshold = sum(
                (model.state_dict()[name].abs() < threshold).sum().item()
                for name in names)
            group_values.append((group, names, num_values_below_threshold))
    elif p_method == 'minimum_weight':
        for group, names in groups.items():
            min_weight = min((model.state_dict()[name].pow(2).sum() /
                              model.state_dict()[name].numel()).item()
                             for name in names)
            group_values.append((group, names, min_weight))

    # 根据第三个元素（即计数或权重）对group_values列表进行排序，然后选择前top_p个
    sorted_groups = sorted(
        group_values,
        key=lambda x: x[2],
        reverse=(p_method != 'minimum_weight'))[:top_p]

    # 返回前top_p个组的信息，每个元素是一个元组(group, names, value)
    return sorted_groups


def set_weights_to_zero_and_untrainable(groups, model):
    for group_info in groups:
        _, names, _ = group_info
        for name in names:
            # 获取权重
            weights = model.state_dict()[name]
            # 将权重设置为全 0
            weights.zero_()
            # 将修改后的权重重新赋值给模型中的对应模块
            name = re.sub(r'\.(\d+)', r'[\1]', name)
            exec(
                'model.' + name +
                ' = Parameter(data=weights, requires_grad=False)', globals(),
                locals())


def prune_model(model,
                task_name='',
                opts=['lora'],
                p_method='zeros',
                top_p=1,
                print_names=False):
    # 获取模型的可训练参数名
    names = get_trainable_parameters(model)
    # 根据前缀分组参数
    groups = group_parameters_by_prefix(
        names, opts=opts, print_names=False, task_name=task_name)
    # 找到最适合剪枝的参数组
    sorted_groups = find_group_with_most_small_values(groups, model, p_method,
                                                      top_p)
    # 将找到的参数组的权重设置为0并设为不可训练
    set_weights_to_zero_and_untrainable(sorted_groups, model)
    # 打印剪枝后的信息，可选
    if print_names:
        for group_info in sorted_groups:
            group, _, small_values = group_info
            logger.info(
                f"Pruned group: {group}, with {small_values} small values.")
