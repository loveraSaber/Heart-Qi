import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ 缩放的点积注意力（支持双向掩码） """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature  
        attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm(x)
        q = self.w_qs(x)
        k = self.w_ks(x)
        v = self.w_vs(x)
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        # print(f"mask_shape:{mask.shape}")
        # print(f"mask:{mask}")
        mask = mask.unsqueeze(1).expand(-1, len_q, -1)
        mask = mask.repeat(n_head, 1, 1)  
        output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual

        return output

class Mlp(nn.Module):
    """Transformer层中的MLP(前馈神经网络)"""
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(Mlp, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        
        # 定义MLP层
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        residual = x
        # 先归一化输入
        normalized_input = self.layer_norm(x)
        intermediate_output = F.gelu(self.dense1(normalized_input))
        layer_output = self.dense2(intermediate_output)
        layer_output = self.dropout(layer_output)
        # 残差连接
        return layer_output + residual  # 返回残差结果，无需再次归一化

class TransformerLayer(nn.Module):
    """单个Transformer层,包含自注意力和MLP"""
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, d_k, d_v,):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, d_k, d_v, attention_probs_dropout_prob)
        self.mlp = Mlp(hidden_size, intermediate_size, hidden_dropout_prob)
    def forward(self, layer_input, mask):
        # 通过注意力层
        attention_output = self.attention(layer_input, mask)
        # 通过MLP层
        layer_output = self.mlp(attention_output)
        return layer_output


class Transformer(nn.Module):
    def __init__(self, 
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_hidden_layers,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                d_k, 
                d_v,
            ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                d_k, 
                d_v,
            ) 
            for _ in range(num_hidden_layers)
        ])
        
    def forward(self, layer_input, mask):

        for layer in self.layers:
            layer_input = layer(layer_input, mask)
        return layer_input