import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import math
import numpy as np
from model.torch_wavelets import DWT_2D, IDWT_2D
from model.torch_wavelets_1D import DWT_1D, IDWT_1D
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .linear_transformers.attention.linear_attention import LinearAttention
from .linear_transformers.masking import FullMask, LengthMask



class Spectral_Attention_Structured_Mesh_2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, sr_ratio=1, is_filter=True):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor

        self.w1 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size))

        dim = self.hidden_size
        self.num_heads = num_blocks
        head_dim = dim // num_blocks 
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.is_filter = is_filter

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//4),
        )
        if self.is_filter:
            self.filter = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
                nn.BatchNorm2d(dim),
            )
        
        self.qkv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3)
        )
        self.proj = nn.Linear(dim+dim//4, dim)
        
        self.merge_linear = nn.Linear(dim+dim, dim)
        
        self.apply(self._init_weights)
        self.inner_attention = LinearAttention(dim)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x_ori = x
        B, N, C = x.shape 
        
        ####################### Fourier Attention ##############################
        x = x.reshape(B, H, W, C) 
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = W // 2 + 1  
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1) 
        x = torch.view_as_complex(x).reshape(B, x.shape[1], x.shape[2], C) 
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x_fft = x + x_ori
        
        ####################### Wavelet Attention ##############################
        x = x_ori.view(B, H, W, C).permute(0, 3, 1, 2) 
        x_dwt = self.dwt(self.reduce(x)) 
        Bd, Cd, Hd, Wd = x_dwt.shape 
        if self.is_filter:
            x_dwt = self.filter(x_dwt) 
        
        attn_mask = FullMask(Hd*Wd, device=x.device)
        length_mask = LengthMask(x_dwt.new_full((B,), Hd*Wd, dtype=torch.int64))
        kv = x_dwt.reshape(B, C, -1).permute(0, 2, 1) 
        kv = self.qkv(kv).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = kv[0], kv[1], kv[2]
        x_dwt_attn = self.inner_attention(
            q, 
            k,
            v,
            attn_mask,
            length_mask,
            length_mask
        ).view(B, N, -1).reshape(B, Hd, Wd, C).permute(0, 3, 1, 2) 
        
        x_idwt = self.idwt(x_dwt_attn)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)  
        x_wave = self.proj(torch.cat([x_ori, x_idwt], dim=-1)) 
        
#########################  Gated Fusion #####################################
        x_merged = torch.cat([x_fft, x_wave], dim=-1)
        weight_map = F.sigmoid(self.merge_linear(x_merged))
        x_final_output = weight_map*x_fft + (1-weight_map)*x_wave
        
        return x_final_output





class Spectral_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, sr_ratio=1, is_filter=False):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor

        self.w1 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor)) 
        self.b1 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size))

        dim = self.hidden_size
        self.num_heads = num_blocks
        head_dim = dim // num_blocks 
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.is_filter = is_filter

        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')
        
        self.reduce = nn.Sequential(
            nn.Conv1d(dim, dim//2, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(inplace=True),
        )
        if self.is_filter:
            self.filter = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
                nn.BatchNorm1d(dim),
            )
        self.qkv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3)
        )
        self.proj = nn.Linear(dim+dim//2, dim)
        
        
        self.merge_linear = nn.Linear(dim+dim, dim)
                
        self.apply(self._init_weights)
        self.inner_attention = LinearAttention(dim)                                           
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x): 
        x_ori = x
        B, N, C = x.shape 
        
        ####################### Fourier Attention ##############################
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, x.shape[1], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1  
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1) 
        x = torch.view_as_complex(x).reshape(B, x.shape[1], C)
        x = torch.fft.irfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N, C)
        x_fft = x + x_ori
        
        ####################### Wavelet Attention ##############################
        x = x_ori.permute(0, 2, 1) 
        x_dwt = self.dwt(self.reduce(x)) 
        Bd, Cd, Nd = x_dwt.shape 
        if self.is_filter:
            x_dwt = self.filter(x_dwt) 
        
        attn_mask = FullMask(Nd, device=x.device)
        length_mask = LengthMask(x.new_full((B,), Nd, dtype=torch.int64)) 
        kv = x_dwt.permute(0, 2, 1) 
        kv = self.qkv(kv).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = kv[0], kv[1], kv[2]
        x_dwt_attn = self.inner_attention(
            q, 
            k,
            v,
            attn_mask,
            length_mask,
            length_mask
        ).view(B, Nd, -1).permute(0, 2, 1)  
        
        x_idwt = self.idwt(x_dwt_attn)
        x_idwt = x_idwt.transpose(1, 2) 
        x_wave = self.proj(torch.cat([x_ori, x_idwt], dim=-1))
        
#########################  Gated Fusion #####################################
        x_merged = torch.cat([x_fft, x_wave], dim=-1)
        weight_map = F.sigmoid(self.merge_linear(x_merged))
        x_final_output = weight_map*x_fft + (1-weight_map)*x_wave
        
        return x_final_output
