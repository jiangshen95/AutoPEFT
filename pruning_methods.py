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


def group_parameters_by_prefix(names,
                               opts=[],
                               print_names=False,
                               task_name='',
                               model_name=''):
    """Group parameters based on prefixes"""
    if model_name == 'roberta':
        v_name = 'query.'
        q_name = 'value.'
    else:
        v_name = 'q_proj.'
        q_name = 'v_proj.'
    groups = {}
    names = [
        name for name in names
        if task_name in name and 'head' not in name and any(
            opt in name for opt in opts)
    ]
    for name in names:
        prefix = name.split(task_name)[0]
        prefix = prefix.replace(v_name, '').replace(q_name, '')
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
    prefix = 'module.'
    if p_method == 'zeros':
        for group, names in groups.items():
            total_params = sum(
                model.state_dict()[name].numel() for name in names)
            num_zeros = sum(
                (model.state_dict()[name] == 0).sum().item() for name in names)
            ratio_zeros = num_zeros / total_params if total_params > 0 else 0
            group_values.append((group, names, ratio_zeros))

    elif p_method == 'values_below_threshold':
        threshold = 0.000001  # Adjust the threshold as needed
        for group, names in groups.items():
            total_params = sum(
                model.state_dict()[name].numel() for name in names)
            num_values_below_threshold = sum(
                (model.state_dict()[name].abs() < threshold).sum().item()
                for name in names)
            ratio_below_threshold = num_values_below_threshold / total_params if total_params > 0 else 0
            group_values.append((group, names, ratio_below_threshold))

    elif p_method == 'minimum_weight':
        for group, names in groups.items():
            total_params = sum(
                model.state_dict()[name].numel() for name in names)
            min_weight = min((model.state_dict()[name].pow(2).sum() /
                              model.state_dict()[name].numel()).item()
                             for name in names)
            avg_min_weight = min_weight / total_params if total_params > 0 else 0
            group_values.append((group, names, avg_min_weight))

    elif p_method == 'gradient':
        if gradients is None:
            raise ValueError(
                "Gradients must be provided for 'gradient' p_method")
        for group, names in groups.items():
            total_params = sum(
                model.state_dict()[name].numel() for name in names)
            total_gradient = sum(
                np.abs(gradients[prefix + name]).sum().item() for name in names)
            avg_gradient = total_gradient / total_params if total_params > 0 else 0
            group_values.append((group, names, avg_gradient))

    elif p_method == 'activation':
        if activations is None:
            raise ValueError(
                "Activations must be provided for 'activation' p_method")
        for group, names in groups.items():
            activation_scores = []
            total_params = sum(
                model.state_dict()[name].numel() for name in names)
            for name in names:
                activation = activations[prefix + name]
                activation_score = activation.mean().item()
                activation_scores.append(activation_score)
                break
            avg_activation_score = sum(
                activation_scores) / total_params if total_params > 0 else 0
            group_values.append((group, names, avg_activation_score))

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
            weight = model.state_dict()[
                name[7:]].cpu().numpy()  # Convert weight to numpy array
            norm_gradients[name] = grad * weight

        # Sum normalized gradients for each group
        for group, names in groups.items():
            total_params = sum(
                model.state_dict()[name].numel() for name in names)
            total_snip = sum(
                norm_gradients[prefix + name].sum().item() for name in names)
            avg_snip = total_snip / total_params if total_params > 0 else 0
            group_values.append((group, names, avg_snip))

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
    modelname = model.config.model_type
    names = get_trainable_parameters(model)
    groups = group_parameters_by_prefix(
        names,
        opts=opts,
        print_names=print_names,
        task_name=task_name,
        model_name=modelname)
    sorted_groups = find_group_with_most_small_values(groups, model, p_method,
                                                      top_p, gradients,
                                                      activations)
    # remove_layers(sorted_groups, model)
    if print_names:
        for group_info in sorted_groups:
            group, _, small_values = group_info
            print(f"Pruned group: {group}, with {small_values} small values.")
    match = re.search(r'layer(s)?\.(\d+)', sorted_groups[0][0])
    if match:
        layer_number = match.group(2)
    else:
        layer_number = -1
    if 'lora' in sorted_groups[0][0]:
        layer_type = 'LORA'
    else:
        layer_type = 'ADAPTER'
    return layer_number, layer_type
