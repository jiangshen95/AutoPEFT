import torch

def compute_hessians_and_gradients(model, loss_fn, data, target, epoch = 3):
    """
    Compute the diagonal of the Hessian for each parameter in the model, record gradients, and activation values.
    
    Parameters:
    - model: The neural network model.
    - loss_fn: The loss function.
    - data: Input data.
    - target: Target labels.
    
    Returns:
    - hessians: A dictionary containing the diagonal of the Hessian for each parameter.
    - gradients: A dictionary containing the gradients for each parameter.
    - activations: A dictionary containing the activations for each layer.
    """
    for i in range(epoch):
        model.zero_grad()  # Zero gradients of the model
        activations = {}
        def save_activation(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
            #print(f"Activation for {module}: {activations[module]}")

        # Register hooks for each layer
        # 为每个层注册钩子
        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(save_activation(name)))
            
        for name, module in model.named_modules():
            print(f"Module: {name}")
            for param_name, param in module.named_parameters(recurse=False):
                print(f"  Parameter: {param_name}, Shape: {param.shape}")
                
        output = model(data)  # Forward pass
        print(f"output {output}")
        loss = loss_fn(output, target)  # Compute loss
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)  # First-order gradients
        loss.backward(retain_graph=True)
        hessians = {}
        gradients = {}
        
        for i, (name, param) in enumerate(model.named_parameters()):
            grad = grads[i]
            gradients[name] = gradients.setdefault(
                            name, 0) + grad  # Record gradient
            hessian_diag = []
            
            for j in range(grad.numel()):
                grad2 = torch.autograd.grad(grad.flatten()[j], param, retain_graph=True)[0]
                hessian_diag.append(grad2.flatten()[j].item())
            
            hessians[name] = hessians.setdefault(
                            name, 0) + torch.tensor(hessian_diag).reshape(param.size())
            
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return hessians, gradients, activations

# Define a simple model with three linear layers and ReLU activations
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(5, 3)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(3, 1)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel()
loss_fn = torch.nn.MSELoss()
data = torch.randn(1, 10)  # Batch of 1 sample, each with 10 features
target = torch.randn(1, 1)  # Batch of 1 target value

# Compute the Hessians and gradients
hessians, gradients, activations = compute_hessians_and_gradients(model, loss_fn, data, target)

# # Print Hessians
# for name, hessian in hessians.items():
#     print(f"Hessian diagonal for {name}: {hessian}")

# # Print gradients
# for name, gradient in gradients.items():
#     print(f"Gradient for {name}: {gradient}")

# Print activations
for module, activation in activations.items():
    print(f"Activation for {module}: {activation}")
