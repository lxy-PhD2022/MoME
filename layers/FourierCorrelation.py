import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierBlock(nn.Module):
    def __init__(self, context_window, c_in):
        super().__init__()
        self.feature_size = c_in
        self.seq_length = context_window
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.r2 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.feature_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.feature_size, self.feature_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.feature_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.feature_size))        

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, L, C]
        x = torch.fft.rfft(x, dim=1, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=1, norm="ortho")
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        # o1_real = torch.zeros([B, dimension // 2 + 1, nd],
        #                       device=x.device)
        # o1_imag = torch.zeros([B, dimension // 2 + 1, nd],
        #                       device=x.device)

        o1_real = F.relu(
            torch.einsum('blc,cc->blc', x.real, r) - \
            torch.einsum('blc,cc->blc', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('blc,cc->blc', x.imag, r) + \
            torch.einsum('blc,cc->blc', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # [B, N, T, D]
        x = self.MLP_temporal(x.permute(0, 2, 1), B, N, T)
        return x.permute(0, 2, 1)
    