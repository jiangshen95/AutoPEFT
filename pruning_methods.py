'''

'''


def get_trainable_parameters(model):
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name)
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
            print(f"{prefix}:")
            for name in names:
                print(f"  {name}")
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


def remove(path, model, methods):
    path = path.split('.')
    print(path)
    module = model
    for part in path:  # 遍历到最后一个之前的路径部分，定位到父模块
        if part.isdigit():
            part = int(part)  # 如果是数字，转换为整数索引
            module = module[part]
        elif part == methods:
            module = delattr(module, part)
            break
        else:
            module = getattr(module, part)
        print(part)


def remove_layers(groups, model):
    for group_info in groups:
        group_name, names, _ = group_info
        # 解析层的路径，以便能够定位并删除
        if 'lora' in group_name:
            for name in names:
                if 'lora_B' in name:
                    continue  # 如果路径中包含 'lora_B'，则跳过这个名字的处理
                remove(name, model, 'loras')
        elif 'adapter' in group_name:
            remove(group_name[:-1], model, 'adapters')


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
        names, opts=opts, print_names=print_names, task_name=task_name)
    # 找到最适合剪枝的参数组
    sorted_groups = find_group_with_most_small_values(groups, model, p_method,
                                                      top_p)
    print(sorted_groups)
    # 将找到的参数组的权重设置为0并设为不可训练
    remove_layers(sorted_groups, model)
    # 打印剪枝后的信息，可选
    if print_names:
        for group_info in sorted_groups:
            group, _, small_values = group_info
            print(f"Pruned group: {group}, with {small_values} small values.")
