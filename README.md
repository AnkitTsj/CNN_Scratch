# Custom Neural Network with Forward and Backward Propagation Validation

This repository contains the implementation of a custom Convolutional Neural Network (CNN) in PyTorch, along with methods to validate its forward propagation, backward propagation, 
and end-to-end training workflow. The aim is to ensure that the custom methods for forward and backward passes are correctly implemented and produce results consistent with PyTorch's built-in autograd and 
optimization features.

---

## Features

1. **Custom Forward Propagation**:
   - Includes a custom implementation of the forward pass using PyTorch tensors.

2. **Custom Backward Propagation**:
   - Implements gradient computation manually for various layers and loss functions.

3. **Validation Functions**:
   - `validate_forward`: Compares the custom forward pass outputs with PyTorch’s autograd-backed outputs.
   - `validate_backward`: Verifies the correctness of gradients computed by the custom backward pass.
   - `train_with_optimizer`: Demonstrates the complete training process using PyTorch's optimizer and custom forward/backward methods.

4. **Dataset and Training**:
   - Utilizes the CIFAR-10 dataset for training.
   - Training is demonstrated with a basic CNN architecture.

---

## Files

### 1. `cnn.py`
- Contains the implementation of the custom CNN architecture.
- Defines the forward and backward propagation logic for the network.

### 2. `loss.py`
- Defines the `Loss` class, which encapsulates custom loss computation and gradient calculation for Cross-Entropy (CE), Mean Squared Error (MSE), and Binary Cross-Entropy (BCE).

### 3. `validation.py`
- Implements the validation functions:
  - `validate_forward`: Ensures custom forward propagation matches PyTorch outputs.
  - `validate_backward`: Compares custom gradients with PyTorch autograd gradients.
  - `train_with_optimizer`: Trains the model using PyTorch’s optimizer with custom methods.

### 4. `train.ipynb`
- Handles the overall training process, including dataset loading, model initialization, and training loop.


## Setup and Installation

### Requirements
- Python 3.10+
- PyTorch 1.10+
- torchvision 0.11+
- numpy 
- matplotlib

### Steps to Use
### **1. Clone this repository:
   ```bash
   git clone https://github.com/AnkitTsj/CNN_Scratch.git
   cd your_working_dir
   ```


2. simply open the run.ipynb on colab as provided and run the cells to try it very simple.
---
## or write the script as described--
### **1: Import the modules and create a model:
```python
from cnn import *
from loss import *
from validation import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = "./data"
train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self,batch_size,device):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1,device = device)
        # conv_params = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'out_channels': 16}
        conv_shape = calculate_output_shape((3,32,32),"conv",layer_params={'kernel_size':3,'stride':1,'padding':1,'out_channels':16})
        self.relu = ReLU()
        self.pool = Pool2d(kernel_size=2, stride=2,pool = "max")
        pool_shape = calculate_output_shape(conv_shape,"maxpool",layer_params={'kernel_size':2,'stride':2,'padding':0})
        self.fc1 = FC(batch_size,pool_shape[0]* pool_shape[1] * pool_shape[2], 10,device = device)
        self.softmax = Softmax()
        self.layers = nn.ModuleList([self.conv1, self.relu, self.pool, self.fc1,self.softmax][::-1])

    def forward(self, x):
        x = (x - x.mean()) / (x.std() + 1e-5)  # explicit normalization
        x,_ = self.conv1.custom_forward(x)
        x = self.relu.custom_forward(x)
        x = self.pool.custom_forward(x)
        x = self.fc1.custom_forward(x)
        x  = self.softmax.custom_forward(x,dim = -1)
        return x,self.layers


model = CNN(batch_size=8,device = device)
criterion = nn.CrossEntropyLoss()
loss_module = Loss(criterion)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
### **2. Validate Backward Propagation**
To compare custom backward gradients with PyTorch autograd:
```python
from validation import validate_backward

sample_input = torch.randn(4, 3, 32, 32).to(device)  # Replace dimensions as per your network
sample_target = torch.randint(0, 10, (4,)).to(device)  # Replace target dimensions as needed
validate_backward(model, sample_input, sample_target, nn.CrossEntropyLoss())
```

### **3. Train with Optimizer**
To train the network and validate the complete workflow:
```python
from validation import train_with_optimizer

losses = train_with_optimizer(model, train_loader, nn.CrossEntropyLoss(), num_epochs=5)
```

---

## Results
- The `validate_forward` and `validate_backward` functions ensure the correctness of custom methods.
- Loss values during training can be plotted using the `plot_loss_curve` utility for visualization.


---

## Resources
- CIFAR-10 Dataset: For the dataset used in this project.

