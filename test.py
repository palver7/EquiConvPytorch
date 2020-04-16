import torch
from torch.nn import ZeroPad2d as zeropad
from equi_conv import EquiConv2d, equi_conv2d

# testing the EquiConv2d function
a = torch.randn(2,64,32,32)
convol = EquiConv2d(64,64,3,padding=1, bias=False)
#offset = torch.randn(2,18,32,32)
b = convol(a)
print(b.shape)

# testing the equi_conv2d function
in1 = torch.tensor([[[[1.,1.,2.,2.],[1.,1.,2.,2.],[3.,3.,4.,4.],[3.,3.,4.,4.]]]]).cuda()
#print(in1)
#offset = torch.randn(1,8,4,4).cuda()
samepad = zeropad((2,2,2,2)).cuda()
in2 = samepad(in1)
weights = torch.ones((1,1,2,2)).cuda()
#print(weights)
out1 = equi_conv2d(in2,weights,stride = 2).cuda()
print(out1.shape)
#print(out1)