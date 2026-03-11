import torch
import torch.nn as nn
import numpy as np
import math


class Conv1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv1, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x
    


class Conv2(nn.Module):
    def __init__(
        self,
        in_channels,      
        out_channels,    
        kernel_size=3,   
        stride=1,         
        padding=1,        
        dilation=1,       
        bias=True,        
        w_init="linear",  
    ):
        super(Conv2, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        
        if w_init == "xavier":
            nn.init.xavier_uniform_(self.conv.weight)
        elif w_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)
    

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ 正弦位置编码表 """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


