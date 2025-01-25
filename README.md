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

### 4. `train.py`
- Handles the overall training process, including dataset loading, model initialization, and training loop.

### 5. `utils.py`
- Utility functions for loading data, plotting loss curves, and managing device configurations.

---

## Setup and Installation

### Requirements
- Python 3.10+
- PyTorch 1.10+
- torchvision 0.11+
- numpy 
- matplotlib

### Steps to Install
1. Clone this repository:
   ```bash
   git clone https://github.com/AnkitTsj/CNN_Scratch.git
   cd your_working_dir
   ```


2. Run the training script:
   ```bash
   python run.ipynb
   ```
##OR simply open the run.ipynb on colab as provided and run the cells to try it very simple.
---

## How to Use

### **1. Validate Forward Propagation**
To ensure the correctness of the custom forward pass:
```python
from validation import validate_forward

sample_input = torch.randn(4, 3, 32, 32).to(device)  # Replace dimensions as per your network
validate_forward(model, sample_input)
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

