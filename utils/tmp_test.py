import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a 1x1 convolution layer with identity matrix as weights
conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, bias=False)

# Initialize weights as identity matrix
with torch.no_grad():
    identity_kernel = torch.eye(16).view(16, 16, 1, 1)  # Identity matrix for 1x1 conv
    conv.weight.copy_(identity_kernel)

# Define padding dimensions: (left, right, top, bottom, front, back, depth, ...)
pad_dims = (1, 1, 1, 1, 0, 0, 0, 0)

# Pad the convolution kernel with zeros
padded_kernel = F.pad(conv.weight, pad=pad_dims, mode='constant', value=0)

# Create a random input tensor
input_tensor = torch.rand((16, 16, 24, 24))

# Apply the modified convolution
output_tensor = F.conv2d(input_tensor, padded_kernel, padding=1)

# Check if input and output tensors are identical
are_equal = torch.all(torch.eq(input_tensor, output_tensor)).item()
print(f"Are input and output tensors identical? {are_equal}")
