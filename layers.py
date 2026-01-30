import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True, norm=False):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.ReflectionPad2d(padding),
                  nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, activation=True, norm=False)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, activation=False, norm=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + x)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        B, C, H, W = x.shape
        proj_x = x.view(B, C, H * W)
        proj_x_T = proj_x.permute(0, 2, 1)
        energy = torch.bmm(proj_x, proj_x_T)
        attention = F.softmax(energy, dim=-1)
        out_proj = torch.bmm(attention, proj_x)
        out = out_proj.view(B, C, H, W)
        return self.alpha * out + x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=2, dilation=2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.sigmoid(out)
    
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.c = c
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports 4D tensor.'
        n, c, h, w = x.size()
        assert c == self.c, 'Input channel mismatch.'
        var = torch.var(x, dim=[1, 2, 3], keepdim=True)
        mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x * self.gamma.view(1, c, 1, 1) + self.beta.view(1, c, 1, 1)
        return x

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm2d(channels)
        self.qkv_proj = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv_proj(x_norm).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        
        return x + self.gamma * self.out_proj(out)

class CrossAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)
        K_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)
        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)
        
        attention = torch.matmul(Q_l, K_r_T) * self.scale
        
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r).permute(0, 3, 1, 2)
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l).permute(0, 3, 1, 2)
        
        return x_l + self.beta * F_r2l, x_r + self.gamma * F_l2r