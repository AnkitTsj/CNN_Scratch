# Pad a tensor with specified padding and optional value
import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import Parameter


def pad_tsr(tensor, padding, val=0):
    # Only pad dynamically when necessary
    if padding > 0:
        return F.pad(tensor, (padding, padding, padding, padding), value=val)
    return tensor


# Initialize filters with He initialization for different layer types
def init_filters(size: tuple, device, if_fc=False, if_bs=False):
    # Calculate He initialization standard deviation
    fan_in = size[0]
    he_std = math.sqrt(2.0 / fan_in)

    # Handle different initialization scenarios for fully connected, bias, and conv layers
    if if_fc:
        return torch.rand(size=(size[1], size[0]), requires_grad=True).squeeze(0).squeeze(0).to(
            device=device).half() * he_std

    if if_bs:
        return torch.rand(size=size, requires_grad=True).to(device=device).half() * he_std

    # Handle 2D and 3D tensor initializations with additional dimension handling
    if len(size) == 2:
        return torch.rand(size, requires_grad=True).unsqueeze(0).unsqueeze(0).to(device=device).half() * he_std

    if len(size) == 3:
        size = tuple([1]) + size
        return torch.randn(size=size, requires_grad=True).unsqueeze(0).to(device=device).half() * he_std

    # Default initialization for other tensor sizes
    return torch.randn(size=size, requires_grad=True).to(device=device).half() * he_std


# Custom ReLU activation layer with forward and backward methods
class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.output = None

    # Forward pass: max(x, 0)
    def custom_forward(self, x):
        self.input = x
        return torch.max(x, torch.zeros_like(x))

    # Backward pass: gradient only for positive inputs
    def custom_backward(self, grad_output, if_update):
        grad_input = grad_output.clone()
        grad_input[self.input <= 0] = 0
        return grad_input


# Custom Softmax activation layer with numerical stability
class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.output = None

    # Forward pass with max subtraction for numerical stability
    def custom_forward(self, x, dim):
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x_shifted = torch.exp(x - x_max)
        self.output = exp_x_shifted / torch.sum(exp_x_shifted, dim=dim, keepdim=True)
        return self.output

    # Backward pass using output and gradient
    def custom_backward(self, grad_output, if_update):
        grad_input = self.output * (grad_output - (grad_output * self.output).sum(dim=1, keepdim=True))
        return grad_input


# Custom Sigmoid activation layer
class sigmoid(nn.Module):
    def __init__(self):
        super(sigmoid, self).__init__()
        self.output = None

    # Sigmoid activation: 1 / (1 + exp(-x))
    def custom_forward(self, x):
        self.output = 1 / (1 + torch.exp(-x))
        return self.output

    # Backward pass: gradient based on sigmoid output
    def custom_backward(self, grad_output, if_update):
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input


# Custom 2D Convolution layer with Adam-like optimization
class Conv2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False, lr=0.001,
                 beta1=0.9, beta2=0.999, epsilon=1e-4,device= "cpu"):
        super().__init__()
        # Layer configuration parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Optimization hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize filters and Adam optimization states
        self.filters = Parameter(init_filters((out_channels, in_channels, kernel_size, kernel_size), device=device))
        self.m_weights = torch.zeros_like(self.filters).uniform_(-1e-3, 1e-3)
        self.v_weights = torch.zeros_like(self.filters).uniform_(-1e-3, 1e-3)

        # Optional bias initialization
        if bias:
            self.bias = Parameter(init_filters((out_channels,), device=device))
            self.m_bias = torch.zeros_like(self.bias).uniform_(-1e-3, 1e-3)
            self.v_bias = torch.zeros_like(self.bias).uniform_(-1e-3, 1e-3)
        else:
            self.bias = None
            self.m_bias = None
            self.v_bias = None

        self.t = 0
        self.padded_input = None

    # Convolution forward pass with padding and unfolding
    def custom_forward(self, x):
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        self.padded_input = x_padded.clone()

        # Perform convolution using unfolding and tensor multiplication
        B, C, H, W = x_padded.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1
        x_unfolded = x_padded.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x_unfolded = x_unfolded.contiguous().view(B, C, out_h * out_w, self.kernel_size, self.kernel_size)
        x_unfolded = x_unfolded.permute(0, 2, 1, 3, 4).unsqueeze(1)

        filters = self.filters.unsqueeze(0).unsqueeze(2)
        output = (x_unfolded * filters).sum(dim=[3, 4, 5])

        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        output = output.view(B, self.out_channels, out_h, out_w)
        return output, self.filters

    # Backward pass with Adam-like weight updates
    def custom_backward(self, grad_output, if_update):
        # Compute gradients for filters and input
        grad_filters = F.conv2d(
            self.padded_input.permute(1, 0, 2, 3),
            grad_output.permute(1, 0, 2, 3),
            stride=self.stride,
        )
        grad_filters = grad_filters.permute(1, 0, 2, 3)
        grad_filters = torch.clamp(grad_filters, -1.0, 1.0)

        # Compute input gradient
        grad_input = F.conv2d(grad_output, self.filters.permute(1, 0, 2, 3).flip([2, 3]), bias=None, stride=self.stride,
                              padding=self.padding)

        # Compute bias gradient if bias is present
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if self.bias is not None else None

        # Perform Adam-like weight updates
        if if_update:
            self.t += 1
            # Update weights using Adam-like rule
            self.m_weights = (self.beta1 * self.m_weights) + ((1 - self.beta1) * grad_filters)
            self.v_weights = (self.beta2 * self.v_weights) + ((1 - self.beta2) * (grad_filters ** 2))
            m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
            v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
            self.filters.data -= (self.lr * (m_hat_weights / (torch.sqrt(v_hat_weights) + self.epsilon)))

            # Update bias using similar Adam-like rule if bias exists
            if self.bias is not None:
                grad_bias = torch.clamp(grad_bias, -1.0, 1.0)
                self.m_bias = (self.beta1 * self.m_bias) + ((1 - self.beta1) * grad_bias)
                self.v_bias = (self.beta2 * self.v_bias) + ((1 - self.beta2) * (grad_bias ** 2))
                m_hat_bias = self.m_bias / (1 - self.beta1 ** self.t)
                v_hat_bias = self.v_bias / (1 - self.beta2 ** self.t)
                self.bias.data -= (self.lr * (m_hat_bias / (torch.sqrt(v_hat_bias) + self.epsilon)))

        grad_input = torch.clamp(grad_input, -1.0, 1.0)
        return grad_input


# Pooling layer supporting max and average pooling
class Pool2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, pool="avg"):
        super().__init__()
        # Layer configuration parameters
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_indices = None
        self.in_channels = in_channels
        self.pool = pool
        self.input = None
        self.type_ = pool

    # Forward pass for pooling operation
    def custom_forward(self, x):
        self.input = x.clone()
        B, C, H, W = x.shape

        # Calculate output dimensions
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        S_b = ((H - self.kernel_size) // self.stride + 1) * ((W - self.kernel_size) // self.stride + 1)

        # Unfold input tensor
        self.unfolded_input = x.shape
        x = x.unfold(2, self.kernel_size, self.stride)
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.contiguous().view(B, C, S_b, self.kernel_size, self.kernel_size)

        # Perform max or average pooling
        if self.pool == "max":
            x = x.view(B, C, S_b, -1)
            x, self.max_indices = torch.max(x, dim=-1)
            self.type_ = "max"
        elif self.pool == "avg":
            x = x.sum(dim=[3, 4]) / (self.kernel_size ** 2)
            self.type_ = "avg"
        else:
            raise ValueError("pool_type must be 'max' or 'avg'")

        x = x.view(B, C, out_h, out_w)
        return x

    # Backward pass for pooling layer
    def custom_backward(self, grad_output, if_update):
        # Handling max pooling gradient
        if self.type_ == "max":
            unfolded_input = torch.zeros(self.unfolded_input).to(self.input.device).half().uniform_(-1e-3, 1e-3)
            max_idx_mask = self.max_indices.view(unfolded_input.shape[0], unfolded_input.shape[1],
                                                 unfolded_input.shape[2], -1)
            grad_output = grad_output.view(unfolded_input.shape[0], unfolded_input.shape[1], unfolded_input.shape[2],
                                           -1).half()
            unfolded_input.scatter(dim=-1, index=max_idx_mask, src=grad_output)
            grad_input = unfolded_input

        # Handling average pooling gradient
        elif self.type_ == "avg":
            # Handle different input dimensionality
            len_fim = len(grad_output.shape)
            if len_fim == 2:
                grad_output = grad_output.unsqueeze(0).unsqueeze(0)
            if len_fim == 3:
                grad_output = grad_output.unsqueeze(0)

            # Interpolate gradient and normalize
            grad_input = F.interpolate(
                grad_output,
                scale_factor=self.kernel_size,
                mode="nearest"
            ) / (self.kernel_size ** 2)
        else:
            raise ValueError("pool_type must be 'max' or 'avg'")

        # Reshape and clamp gradient
        grad_input = grad_input.view(self.input.shape)
        grad_input = torch.clamp(grad_input, -1.0, 1.0)
        return grad_input


# Fully Connected layer with Adam-like optimization
class FC(nn.Module):
    def __init__(self, batch_size, out_channels, units, bias=False, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-4,device = "cpu"):
        super().__init__()
        # Layer configuration parameters
        self.out_channels = out_channels
        self.units = units

        # Initialize weights with He initialization
        self.weights = init_filters((out_channels, units), if_fc=True, device=device)

        # Optimization hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize Adam-like optimization states
        self.m_weights = torch.zeros_like(self.weights).uniform_(-1e-3, 1e-3)
        self.v_weights = torch.zeros_like(self.weights).uniform_(-1e-3, 1e-3)

        # Handle optional bias
        self.if_bias = bias
        if self.if_bias is not False:
            self.if_bias = True
            self.bias = init_filters((units, 1), if_bs=True, device=device)
            self.m_bias = torch.zeros_like(self.bias).uniform_(-1e-3, 1e-3)
            self.v_bias = torch.zeros_like(self.bias).uniform_(-1e-3, 1e-3)
        else:
            self.if_bias = False
            self.bias = torch.zeros((units, 1)).uniform_(-1e-3, 1e-3)
            self.m_bias = None
            self.v_bias = None

        self.t = 0
        self.batch_sized = batch_size
        self.input = None

    # Forward pass: matrix multiplication with optional bias
    def custom_forward(self, x):
        self.input = x.clone()
        x = x.view(x.size(0), -1)
        self.weights = self.weights.view(self.units, -1).contiguous()
        output = x @ self.weights.T + self.bias.view(1, self.units)
        return output

    # Backward pass with Adam-like weight updates
    def custom_backward(self, grad_output, if_update):
        # Compute gradients for weights and bias
        grad_weights = grad_output.T @ self.input.view(grad_output.shape[0], -1)
        grad_weights = grad_weights.view(-1, self.input.shape[1], self.input.shape[2], self.input.shape[3])
        grad_bias = torch.sum(grad_output, dim=0, keepdim=True).T

        # Clamp gradients
        self.grad_weights = torch.clamp(grad_weights, -1.0, 1.0)

        # Perform Adam-like weight updates
        if if_update == 1:
            self.t += 1

            # Update weights
            self.m_weights = self.m_weights.view(grad_weights.shape)
            self.v_weights = self.v_weights.view(grad_weights.shape)
            self.m_weights = (self.beta1 * self.m_weights) + ((1 - self.beta1) * grad_weights)
            self.v_weights = (self.beta2 * self.v_weights) + ((1 - self.beta2) * (grad_weights ** 2))
            m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
            v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)

            self.weights = self.weights.view(grad_weights.shape)
            self.weights = self.weights - (self.lr * (m_hat_weights / (torch.sqrt(v_hat_weights) + self.epsilon)))
            self.weights = self.weights.view(grad_weights.shape[0], -1)

            # Update bias if present
            if self.if_bias is not False:
                grad_bias = torch.clamp(grad_bias, -1.0, 1.0)
                self.m_bias = (self.beta1 * self.m_bias) + ((1 - self.beta1) * grad_bias)
                self.v_bias = (self.beta2 * self.v_bias) + ((1 - self.beta2) * (grad_bias ** 2))
                m_hat_bias = self.m_bias / (1 - self.beta1 ** self.t)
                v_hat_bias = self.v_bias / (1 - self.beta2 ** self.t)
                self.bias = self.bias - (self.lr * (m_hat_bias / (torch.sqrt(v_hat_bias) + self.epsilon)))

        # Compute input gradient
        grad_input = grad_output @ self.weights
        return grad_input


def calculate_output_shape(input_shape, layer_type, layer_params):
    """
    Calculates the output shape of a layer based on its type and parameters.

    Supports calculation for conv, maxpool, avgpool, linear, and flatten layers.
    Handles various input parameter formats and provides robust shape computation.
    """
    # Convolution layer shape calculation
    if layer_type == 'conv':
        # Validate required parameters
        if not all(key in layer_params for key in ['kernel_size', 'stride', 'padding']):
            raise ValueError("Conv layer requires 'kernel_size', 'stride', and 'padding' parameters.")

        # Normalize parameter formats to tuples
        kernel_size = layer_params['kernel_size'] if isinstance(layer_params['kernel_size'], tuple) else (
        layer_params['kernel_size'], layer_params['kernel_size'])
        stride = layer_params['stride'] if isinstance(layer_params['stride'], tuple) else (
        layer_params['stride'], layer_params['stride'])
        padding = layer_params['padding'] if isinstance(layer_params['padding'], tuple) else (
        layer_params['padding'], layer_params['padding'])
        dilation = layer_params.get('dilation', (1, 1)) if isinstance(layer_params.get('dilation', 1), tuple) else (
        layer_params.get('dilation', 1), layer_params.get('dilation', 1))

        # Calculate output dimensions
        in_channels, in_height, in_width = input_shape
        out_height = math.floor((in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        out_width = math.floor((in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        out_channels = layer_params.get('out_channels', in_channels)
        return (out_channels, out_height, out_width)

    # Pooling layer shape calculation
    elif layer_type in ('maxpool', 'avgpool'):
        # Validate required parameters
        if not all(key in layer_params for key in ['kernel_size', 'stride', 'padding']):
            raise ValueError(f"{layer_type} layer requires 'kernel_size', 'stride', and 'padding' parameters.")

        # Normalize parameter formats to tuples
        kernel_size = layer_params['kernel_size'] if isinstance(layer_params['kernel_size'], tuple) else (
        layer_params['kernel_size'], layer_params['kernel_size'])
        stride = layer_params['stride'] if isinstance(layer_params['stride'], tuple) else (
        layer_params['stride'], layer_params['stride'])
        padding = layer_params['padding'] if isinstance(layer_params['padding'], tuple) else (
        layer_params['padding'], layer_params['padding'])

        # Calculate output dimensions
        in_channels, in_height, in_width = input_shape
        out_height = math.floor((in_height + 2 * padding[0] - (kernel_size[0] - 1) - 1) / kernel_size[0] + 1)
        out_width = math.floor((in_width + 2 * padding[1] - (kernel_size[1] - 1) - 1) / kernel_size[1] + 1)
        return (in_channels, out_height, out_width)

    # Linear layer shape calculation
    elif layer_type == 'linear':
        in_features = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        out_features = layer_params['out_features']
        return (out_features,)

    # Flatten layer shape calculation
    elif layer_type == 'flatten':
        if len(input_shape) != 3:
            raise ValueError("Flatten layer expects 3D input (C, H, W)")
        in_channels, in_height, in_width = input_shape
        return (in_channels * in_height * in_width,)

    # Invalid layer type
    else:
        return None


# Custom loss computation module
class Loss(nn.Module):
    def __init__(self, loss_fxn):
        super().__init__()
        # Store loss function and related attributes
        self.output = None
        self.target = None
        self.loss = None
        self.loss_fxn = loss_fxn

    def forward(self, output, target):
        """
        Compute the loss and store the input and target for backward computation.
        """
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
    loss = loss_module.forward(output, target)

    # Generate loss gradients
    grad_loss = loss_module.backward(type_=loss_type)

    # Backpropagate through layers
    grads = layer_back(grad_loss, layers)
    del grads

    return loss

