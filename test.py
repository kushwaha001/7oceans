import torch
from mamba_ssm.modules.mamba2 import Mamba2                                                                                                                                                              
m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2).cuda()                                                                                                                                           
x = torch.randn(1, 100, 256).cuda()
y = m(x)                                                                                                                                                                                                 
print('Input:', x.shape, 'Output:', y.shape)                                                                                                                                                           
print('All good.') 
