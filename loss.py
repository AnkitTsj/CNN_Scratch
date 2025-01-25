import torch
import torch.nn as nn


# Custom loss computation module
class Loss(nn.Module):
    def __init__(self, loss_fxn):
        super().__init__()
        # Store loss function and related attributes
        self.output = None
        self.target = None
        self.loss = None
        self.loss_fxn = loss_fxn

    def forward(self, output, target,if_softmax = True):
        """
        Compute the loss and store the input and target for backward computation.
        """
        self.output = output[0]
        if if_softmax:
        # Apply Softmax to the output
          self.output = Softmax().custom_forward(output[0], dim=-1)
        self.target = target.clone()

        # Compute loss
        self.loss = self.loss_fxn(self.output, target)
        return self.loss

    def backward(self, type_):
        """
        Compute the gradient of the loss with respect to the input.
        Supports different loss types: MSE, CE (Cross-Entropy), BCE (Binary Cross-Entropy)
        """
        if type_ == "MSE":
            # Mean Squared Error gradient
            grad_input = 2 * (self.output - self.target) / self.input.numel()
            return grad_input

        if type_ == "CE":
            # Cross-Entropy gradient with one-hot encoding
            num_classes = self.output.shape[1]
            target_one_hot = torch.nn.functional.one_hot(self.target, num_classes=num_classes).float()
            grad_input = self.output - target_one_hot
            del target_one_hot
            return grad_input

        if type_ == "BCE":
            # Binary Cross-Entropy gradient
            grad_input = -(self.target / self.input + (1 - self.target) / (1 - self.output))
            return grad_input


# Perform backward pass through layers
def layer_back(loss_grad, layers):
    """
    Propagate gradients through network layers.

    Args:
        loss_grad: Gradient from loss computation
        layers: List of network layers to backpropagate through

    Returns:
        Final gradient after backpropagation
    """
    grad = loss_grad.half()
    for layer in layers:
        grad = layer.custom_backward(grad.half(), if_update=1)
    return grad


# Complete backward pass computation
def backward(criterion, output, target, layers, loss_type, loss_module):
    """
    Perform a complete backward pass through the model.

    Computes loss, generates loss gradients, and backpropagates through layers.
    """
    # Compute loss
    loss = loss_module.forward(output, target,if_softmax = False)

    # Generate loss gradients
    grad_loss = loss_module.backward(type_=loss_type)

    # Backpropagate through layers
    grads = layer_back(grad_loss, layers)
    del grads

    return loss

