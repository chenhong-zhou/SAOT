import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck

class DWT_Function(Function): 
    @staticmethod
    def forward(ctx, x, w_ll, w_hh): 
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_hh) 
        ctx.shape = x.shape 

        dim = x.shape[1] 
        x_ll = torch.nn.functional.conv1d(x, w_ll.expand(dim, -1, -1), stride = 2, groups = dim) 
        x_hh = torch.nn.functional.conv1d(x, w_hh.expand(dim, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_hh], dim=1) 
        return x

    @staticmethod
    def backward(ctx, dx): 
        if ctx.needs_input_grad[0]:
            w_ll, w_hh = ctx.saved_tensors
            B, C, N = ctx.shape 
            dx = dx.view(B, 2, -1, N//2) 

            dx = dx.transpose(1,2).reshape(B, -1, N//2)  
            filters = torch.cat([w_ll, w_hh], dim=0).repeat(C, 1, 1) 
            dx = torch.nn.functional.conv_transpose1d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, N = x.shape
        x = x.view(B, 2, -1, N).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, N)
        filters = filters.repeat(C, 1, 1)
        x = torch.nn.functional.conv_transpose1d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, N = ctx.shape
            C = C // 2
            dx = dx.contiguous()

            w_ll, w_hh = torch.unbind(filters, dim=0) 
            x_ll = torch.nn.functional.conv1d(dx, w_ll.unsqueeze(1).expand(C, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv1d(dx, w_hh.unsqueeze(1).expand(C, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_hh], dim=1) # x_lh, x_hl,
        return dx, None



class IDWT_1D(nn.Module):
    def __init__(self, wave):
        super(IDWT_1D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi) 
        rec_lo = torch.Tensor(w.rec_lo) 
        
        w_ll = rec_lo.unsqueeze(0).unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_hh], dim=0) 
        self.register_buffer('filters', filters) 
        self.filters = self.filters  

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_1D(nn.Module):
    def __init__(self, wave):
        super(DWT_1D, self).__init__()
        w = pywt.Wavelet(wave) 
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])  
        
        self.register_buffer('w_ll', dec_lo.unsqueeze(0).unsqueeze(0)) 
        self.register_buffer('w_hh', dec_hi.unsqueeze(0).unsqueeze(0)) 
        
        self.w_ll = self.w_ll 
        self.w_hh = self.w_hh

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_hh)


