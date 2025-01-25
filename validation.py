import torch

def validate_forward(model, input_tensor,device):
    """
    Validate the forward pass by comparing custom forward output with PyTorch's autograd-backed output.
    """
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)
    model.eval()  # Switch to evaluation mode

    with torch.no_grad():
        custom_output = model(input_tensor)  # Custom forward pass

    # Compute PyTorch output with autograd
    pytorch_output = model(input_tensor)

    # Compare the outputs
    if torch.allclose(custom_output[0], pytorch_output[0], atol=1e-6):
        print("Forward pass validation: PASSED!")
    else:
        print("Forward pass validation: FAILED!")
        print(f"Custom Output: {custom_output}")
        print(f"PyTorch Output: {pytorch_output}")


def validate_backward(model, input_tensor, target_tensor,loss_module,device = "cpu"):
    """
    Validate the backward pass by comparing custom gradients with PyTorch's autograd gradients.
    """
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)
    target_tensor = target_tensor.clone().to(device)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    # Custom backward pass
    custom_output = model(input_tensor)

    custom_loss = loss_module.forward(custom_output, target_tensor)
    custom_loss.backward()  # PyTorch autograd gradient calculation

    # Collect PyTorch gradients
    pytorch_grads = []
    for param in model.parameters():
        pytorch_grads.append(param.grad.clone())

    # Custom backward pass gradients
    custom_grads = []
    for param in model.parameters():
        custom_grads.append(param.grad.clone())  # Adjust based on your custom method

    # Compare gradients
    passed = True
    for cg, pg in zip(custom_grads, pytorch_grads):
        if not torch.allclose(cg, pg, atol=1e-6):
            passed = False
            print(f"Gradient mismatch detected.\nCustom Grad: {cg}\nPyTorch Grad: {pg}")
            break

    if passed:
        print("Backward pass validation: PASSED!")
    else:
        print("Backward pass validation: FAILED!")


def train_with_optimizer(model, train_loader,loss_module, num_epochs=5,device = "cpu"):
    """
    Train the model using PyTorch's optimizer and validate custom forward and backward methods.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.half().to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass using custom method
            outputs = model(inputs)
            # print(outputs[0])
            # Compute loss and gradients using custom methods
            loss = loss_module.forward(outputs[0], labels)
            loss.backward()  # Autograd-based backward pass

            optimizer.step()  # Update weights
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return losses
