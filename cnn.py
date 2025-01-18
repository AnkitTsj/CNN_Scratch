
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

def pad_tsr(tensor, padding, val=0):
    """Pads a tensor on all sides.

    Args:
        tensor (torch.Tensor): The input tensor.
        padding (int): The amount of padding to add to each side.
        val (int, optional): The value to pad with. Defaults to 0.

    Returns:
        torch.Tensor: The padded tensor.
    """
    return torch.constant_pad_nd(tensor, (padding, padding, padding, padding), value=val)

def init_filters(size: tuple, if_fc=False, if_bs=False):
    """Initializes filters (tensors) with random values.

    Args:
        size (tuple): The size of the filter.
        if_fc (bool, optional): If True, initializes for a fully connected layer. Defaults to False.
        if_bs (bool, optional): If True, initializes for a batch size related tensor. Defaults to False.

    Returns:
        torch.Tensor: The initialized filter.
    """
    if if_fc:
        return torch.rand(size=size, requires_grad=True).squeeze(0).squeeze(0)
    if if_bs:
        return torch.rand(size=size, requires_grad=True)
    if len(size) == 2:
        return torch.rand(size, requires_grad=True).unsqueeze(0).unsqueeze(0)
    if len(size) == 3:
        size = tuple([1]) + size  # Add a channel dimension
        return torch.randn(size=size, requires_grad=True).unsqueeze(0) # Add a batch dimension

    return torch.randn(size=size, requires_grad=True)


class Relu(nn.Module):
    """ReLU activation function module."""

    def __init__(self):
        super(Relu, self).__init__()
        self.output = None  # Store the output for backward pass (not strictly necessary with autograd)

    def forward(self, x):
        """Forward pass of ReLU.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying ReLU.
        """
        self.input = x  # Store input for backward pass
        return torch.max(x, torch.zeros_like(x))

    def backward(self, grad_output, type_):
        """Backward pass of ReLU.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.
            type_ (str): Type of backward pass (not used here, but kept for consistency).

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input.
        """
        grad_input = grad_output.clone()
        grad_input[self.input <= 0] = 0  # Zero out gradients for negative inputs
        return grad_input


class Softmax(nn.Module):
    """Softmax activation function module."""

    def __init__(self):
        super(Softmax, self).__init__()
        self.output = None # Store the output for backward pass

    def forward(self, x):
        """Forward pass of Softmax.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying Softmax.
        """
        self.output = torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)
        return self.output

    def backward(self, grad_output, type_):
        """Backward pass of Softmax (using Jacobian).

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.
            type_ (str): Type of backward pass (not used here).

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input.
        """
        batch_size, num_classes = self.output.shape
        grad_input = torch.zeros_like(self.output)
        for i in range(batch_size):
            softmax_output = self.output[i].unsqueeze(1)
            jacobian_matrix = torch.diagflat(softmax_output) - softmax_output @ softmax_output.T
            grad_input[i] = grad_output[i] @ jacobian_matrix
        return grad_input


class sigmoid(nn.Module):
    """Sigmoid activation function module."""

    def __init__(self):
        super(sigmoid, self).__init__()
        self.output = None # Store the output for backward pass

    def forward(self, x):
        """Forward pass of Sigmoid.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying Sigmoid.
        """
        self.output = 1 / (1 + torch.exp(-x))
        return self.output

    def backward(self, grad_output, type_):
        """Backward pass of Sigmoid.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.
            type_ (str): Type of backward pass (not used here).

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input.
        """
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input



class Conv2d(nn.Module):
    """
    Custom 2D Convolution layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Amount of zero-padding to be applied.
        bias: Whether to use bias or not.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False):
        super().__init__()
        self.filters = Parameter(init_filters((out_channels, in_channels, kernel_size, kernel_size)))
        self.D = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.X = None
        if bias:
            self.bias = Parameter(init_filters(size=(out_channels, in_channels, kernel_size, kernel_size)))
        else:
            self.bias = Parameter(torch.zeros(size=(out_channels, in_channels, kernel_size, kernel_size)))
        self.bias_man = False

    def forward(self, x):
        """
        Forward pass of the convolution layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after convolution and a tuple containing the output and filters.
        """
        self.input = x.clone()
        self.X = x.clone()
        x = pad_tsr(x, padding=self.padding)
        B, C, H, W = x.shape
        F_c = self.filters.shape[1]
        F_h = self.kernel_size
        F_w = self.kernel_size
        D = self.D

        # Reshape input for efficient convolution
        self.X = self.X.unfold(1,F_c,self.stride).unfold(2, F_h, self.stride).unfold(3, F_w, self.stride)
        out_h, out_w = self.X.shape[2], self.X.shape[3]
        self.X = self.X.contiguous().view(B, -1, C, F_h, F_w)
        self.X = self.X.repeat(1, D, 1, 1, 1)
        S_b = out_h * out_w
        self.X = self.X.view(self.X.shape[0], -1, S_b, F_c, F_h, F_w)

        # Prepare filters and bias for efficient computation
        filters = self.filters.unsqueeze(1)
        if self.bias_man == False:
            self.broad_bias = self.bias.unsqueeze(1).repeat(1, S_b, 1, 1, 1).unsqueeze(0).repeat(B, 1, 1, 1, 1, 1)
            self.bias_man = True

        # Perform convolution and add bias
        self.X = self.X * filters + self.broad_bias
        self.X = self.X.sum(dim=[3, 4, 5])
        self.X = self.X.view(B, D, out_h, out_w)
        return self.X, self.filters

    def backward(self, grad_output,if_update):
        """
        Backward pass (gradient calculation) for the convolution layer.

        Args:
            grad_output: Gradient of the loss with respect to the output.
            if_update: Flag to determine whether to update filter gradients.

        Returns:
            Gradient of the loss with respect to the input.
        """
        grad_filters = F.conv2d(
            self.padded_input,
            grad_output,
            stride=self.stride,
            padding=self.padding
        )
        if if_update:
            if self.filters.grad is None:
                self.filters.grad = grad_filters
            else:
                self.filters.grad += grad_filters

        grad_input = F.conv_transpose2d(
            grad_output,
            self.filters,
            stride=self.stride,
            padding=self.padding
        )
        if self.bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
            if self.bias.grad is None:
                self.bias.grad = grad_bias
            else:
                self.bias.grad += grad_bias

        return grad_input


class Pool2d(nn.Module):
    """
    Custom 2D Pooling layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the pooling kernel.
        stride: Stride of the pooling operation.
        padding: Amount of zero-padding to be applied.
        pool: Type of pooling ('max' or 'avg').
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, pool="avg"):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.pool = pool
        self.input = None

    def forward(self, x):
        """
        Forward pass of the pooling layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after pooling.
        """
        self.input = x.clone()
        B, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        S_b = ((H - self.kernel_size) // self.stride + 1) * ((W - self.kernel_size) // self.stride + 1)

        x = x.unfold(2, self.kernel_size, self.stride)
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.contiguous().view(B, C, S_b, self.kernel_size, self.kernel_size)

        if self.pool == "max":
            x = x.view(B, C, S_b, -1)
            x, _ = torch.max(x, dim=-1)
        elif self.pool == "avg":
            x = x.sum(dim=[3, 4]) / (self.kernel_size ** 2)
        else:
            raise ValueError("Invalid pool type. Must be 'max' or 'avg'.")

        x = x.view(B, C, out_h, out_w)
        return x

    def backward(self, grad_output, pool_type, if_update):
        """
        Backward pass (gradient calculation) for the pooling layer.

        Args:
            grad_output: Gradient of the loss with respect to the output.
            pool_type: Type of pooling ('max' or 'avg').
            if_update: Flag to determine whether to update gradients (not applicable for pooling).

        Returns:
            Gradient of the loss with respect to the input.
        """
        if pool_type == "max":
            # Implement max pooling backpropagation
            # (This part needs to be corrected)
            unfolded_input = self.unfolded_input  # Access unfolded input (needs to be stored in forward pass)
            max_mask = unfolded_input == unfolded_input.max(dim=1, keepdim=True).values
            grad_input = (max_mask * grad_output.view(grad_output.size(0), 1, -1)).sum(dim=-1)

        elif pool_type == "avg":
            len_fim = len(grad_output.shape)
            if len_fim == 2:
                grad_output = grad_output.unsqueeze(0).unsqueeze(0)
            if len_fim == 3:
                grad_output = grad_output.unsqueeze(0)
            grad_input = F.interpolate(
                grad_output,
                scale_factor=self.kernel_size,
                mode="nearest"
            ) / (self.kernel_size ** 2)
        else:
            raise ValueError("Invalid pool type. Must be 'max' or 'avg'.")

        grad_input = grad_input.view(self.input.shape)
        return grad_input


class FC(nn.Module):
    """
    Custom fully-connected layer with Adam optimizer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        lr: Learning rate for Adam optimizer.
        beta1: Beta1 parameter for Adam optimizer.
        beta2: Beta2 parameter for Adam optimizer.
        epsilon: Epsilon parameter for Adam optimizer.
    """
    def __init__(self, in_channels, out_channels, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = Parameter(init_filters((out_channels, in_channels), if_fc=True))
        self.bias = Parameter(init_filters((out_channels, 1), if_bs=True))

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = torch.zeros_like(self.weights)
        self.v_weights = torch.zeros_like(self.weights)
        self.m_bias = torch.zeros_like(self.bias)
        self.v_bias = torch.zeros_like(self.bias)
        self.t = 0
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        """
        Forward pass of the fully-connected layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after linear transformation.
        """
        self.input = x.clone()
        x = x.view(-1, self.in_channels)
        output = x @ self.weights.T + self.bias.T
        return output

    def backward(self, grad_output, type_, if_update):
        """
        Backward pass (gradient calculation) for the fully-connected layer.
        Performs Adam optimization if update flag is set.

        Args:
            grad_output: Gradient of the loss with respect to the output.
            type_: Not used (for compatibility with Pool2d backward).
            if_update: Flag to determine whether to update weights and bias.

        Returns:
            Gradient of the loss with respect to the input.
        """

        self.grad_weights = grad_output.T @ self.input
        self.grad_bias = torch.sum(grad_output, dim=0, keepdim=True).T
        if if_update == 1:
            self.t += 1
            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * self.grad_weights
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (self.grad_weights ** 2)
            m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
            v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
            self.weights = self.weights - self.lr * m_hat_weights / (
                    torch.sqrt(v_hat_weights) + self.epsilon)
            self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * self.grad_bias
            self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (self.grad_bias ** 2)
            m_hat_bias = self.m_bias / (1 - self.beta1 ** self.t)
            v_hat_bias = self.v_bias / (1 - self.beta2 ** self.t)
            self.bias = self.bias - self.lr * m_hat_bias / (torch.sqrt(v_hat_bias) + self.epsilon)
            self.bias -= self.lr * m_hat_bias / (torch.sqrt(v_hat_bias) + self.epsilon)
        grad_input = grad_output @ self.weights
        return grad_input









