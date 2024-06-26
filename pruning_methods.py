'''

This file contains functions for model pruning, including getting trainable parameters, 
grouping parameters by prefix, finding groups with the smallest values, and more.

'''
import re
import numpy as np


def get_trainable_parameters(model):
    """Get the list of trainable parameter names in the model"""
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
    return names


def group_parameters_by_prefix(names, opts=[], print_names=False, task_name=''):
    """Group parameters based on prefixes"""
    groups = {}
    names = [
        name for name in names
        if task_name in name and 'head' not in name and any(
            opt in name for opt in opts)
    ]
    for name in names:
        prefix = name.split(task_name)[0]
        prefix = prefix.replace('q_proj.', '').replace('v_proj.', '')
        if prefix in groups:
            groups[prefix].append(name)
        else:
            groups[prefix] = [name]
    if print_names:
        for prefix, names in groups.items():
            print(f"{prefix}:")
            for name in names:
                print(f"  {name}")
    return groups


def find_group_with_most_small_values(groups,
                                      model,
                                      p_method,
                                      top_p=1,
                                      gradients=None,
                                      activations=None):
    """Find groups with the smallest values or smallest summed gradients"""
    group_values = []

    if p_method == 'zeros':
        for group, names in groups.items():
            num_zeros = sum(
                (model.state_dict()[name] == 0).sum().item() for name in names)
            group_values.append((group, names, num_zeros))

    elif p_method == 'values_below_threshold':
        threshold = 0.000001  # 可以根据需要调整阈值
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

    elif p_method == 'gradient':
        if gradients is None:
            raise ValueError(
                "Gradients must be provided for 'gradient' p_method")
        for group, names in groups.items():
            total_gradient = sum(
                np.abs(gradients[name]).sum().item() for name in names)
            group_values.append((group, names, total_gradient))

    elif p_method == 'snip':
        if gradients is None:
            raise ValueError("Gradients must be provided for 'snip' p_method")

        # Calculate the gradient magnitudes
        gradient_magnitudes = {}
        for name in gradients:
            gradient_magnitudes[name] = np.abs(gradients[name])

        # Normalize the gradients
        norm_gradients = {}
        for name, grad in gradient_magnitudes.items():
            weight = model.state_dict()[name].cpu().numpy(
            )  # Convert weight to numpy array
            norm_gradients[name] = grad * weight

        # Sum normalized gradients for each group
        for group, names in groups.items():
            total_snip = sum(
                norm_gradients[name].sum().item() for name in names)
            group_values.append((group, names, total_snip))

    elif p_method == 'grasp':
        if gradients is None or hessians is None:
            raise ValueError(
                "Gradients and Hessians must be provided for 'grasp' p_method")

        # Calculate the Hessian-gradient products
        hessian_grad_products = {}
        for name in gradients:
            hessian_grad_products[name] = gradients[name] * hessians[name]

        # Normalize the Hessian-gradient products
        norm_hessian_grad_products = {}
        for name, hg_product in hessian_grad_products.items():
            weight = model.state_dict()[name].cpu().numpy(
            )  # Convert weight to numpy array
            norm_hessian_grad_products[name] = hg_product * weight

        # Sum normalized Hessian-gradient products for each group
        for group, names in groups.items():
            total_grasp = sum(
                norm_hessian_grad_products[name].sum().item() for name in names)
            group_values.append((group, names, total_grasp))

    elif p_method == 'activation':
        if activations is None:
            raise ValueError(
                "Activations must be provided for 'activation' p_method")
        for group, names in groups.items():
            activation_scores = []
            for name in names:
                activation = activations["module." + name]
                # 计算激活值的重要性评分，可以使用不同的统计量，这里用平均值
                activation_score = activation.mean().item()
                activation_scores.append(activation_score)
                break
            total_activation_score = sum(activation_scores)
            group_values.append((group, names, total_activation_score))

    sorted_groups = sorted(
        group_values,
        key=lambda x: x[2],
        reverse=(p_method in ['zeros', 'values_below_threshold']))[:top_p]

    return sorted_groups


def remove(path, model, methods):
    """Remove modules or attributes at the specified path"""
    path = path.split('.')
    module = model
    for part in path:
        if part.isdigit():
            part = int(part)
            module = module[part]
        elif part == methods:
            module = delattr(module, part)
            break
        else:
            module = getattr(module, part)


def remove_layers(groups, model):
    """Remove layers in the specified groups"""
    for group_info in groups:
        group_name, names, _ = group_info
        if 'lora' in group_name:
            for name in names:
                if 'lora_B' in name:
                    continue
                remove(name, model, 'loras')
        elif 'adapter' in group_name:
            remove(group_name[:-1], model, 'adapters')


def prune_model(model,
                task_name='',
                opts=['lora'],
                p_method='zeros',
                top_p=3,
                print_names=False,
                gradients=None,
                hessians=None,
                activations=None):
    """Prune the model's parameters"""
    names = get_trainable_parameters(model)
    groups = group_parameters_by_prefix(
        names, opts=opts, print_names=print_names, task_name=task_name)
    sorted_groups = find_group_with_most_small_values(groups, model, p_method,
                                                      top_p, gradients,
                                                      activations)
    # remove_layers(sorted_groups, model)
    if print_names:
        for group_info in sorted_groups:
            group, _, small_values = group_info
            print(f"Pruned group: {group}, with {small_values} small values.")
    match = re.search('layers.(\d+)', sorted_groups[0][0])
    if match:
        layer_number = match.group(1)
    else:
        layer_number = -1
    if 'lora' in sorted_groups[0][0]:
        layer_type = 'lora'
    else:
        layer_type = 'adapter'
    return layer_number, layer_type
