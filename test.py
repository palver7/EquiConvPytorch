import torch
from torchvision.ops.equi_conv import EquiConv2d
a = torch.randn(2,64,32,32)
convol = EquiConv2d(64,64,3,padding=1)
b = convol(a)
print(b.shape)