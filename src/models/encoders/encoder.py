import torch.nn as nn
from ..frontend.convblock import ConvBlock
# from ..utils.nets_utils import make_non_pad_mask
import math
import torch
from ..encoders.attention import MultiHeadedAttention
from ..encoders.encoder_layer import EncoderLayer
from ..encoders.positionwise_feed_forward import PositionwiseFeedForward
from ..encoders.repeat import repeat
from ..encoders.layer_norm import LayerNorm
from ..encoders.embedding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, config, data_type):
        super().__init__()
        self.data_type = data_type
        if data_type == "mel":
            self.conv = ConvBlock(config.feature_dim, config.reduction_factors)
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(config.feature_dim, config.encoder.hidden_size),           
                PositionalEncoding(config.encoder.hidden_size, config.encoder.hidden_dropout_prob)
            )
        elif data_type == "video":
            self.avg_pooling = nn.AvgPool1d(kernel_size=4, stride=4)
            self.projection = nn.Sequential(
                nn.Conv1d(config.feature_dim // 4, 64, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm1d(config.encoder.hidden_size),
                nn.ReLU(),
                nn.Conv1d(config.encoder.hidden_size, config.encoder.hidden_size, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm1d(config.encoder.hidden_size),
                nn.ReLU()
            )
            self.embed = PositionalEncoding(config.encoder.hidden_size, config.encoder.hidden_dropout_prob)       
        
        elif data_type == "phoneme":
            self.embedding_size = config["encoder"]["hidden_size"]
            n_position = config["max_seq_len"] + 1
            self.embed = torch.nn.Sequential(
                nn.Embedding(config["vocab_size"], self.embedding_size, padding_idx=0),
                PositionalEncoding(config.encoder.hidden_size, config.encoder.hidden_dropout_prob)
            )
            
            
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            config.encoder.hidden_size,
            config.encoder.intermediate_size,
            config.encoder.attention_probs_dropout_prob,
            config.encoder.activation
        )

        self_attn = MultiHeadedAttention
        self_attn_args = (
            config.encoder.num_attention_heads,
            config.encoder.hidden_size,
            config.encoder.attention_probs_dropout_prob
        )
    
        self.encoders = repeat(
            config.encoder.num_hidden_layers,
            lambda lnum: EncoderLayer(
                config.encoder.hidden_size,
                self_attn(*self_attn_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if config.encoder.macaron_style else None,
                None, #不需要卷积神经网络
                config.encoder.attention_probs_dropout_prob,
                config.encoder.normalize_before,
                config.encoder.concat_after
            ),
        )
        if config.encoder.normalize_before:
            self.after_norm = LayerNorm(config.encoder.attention_dim)

    def forward(self, x, x_lens):
        # if self.data_type == "phoneme":
        #     B,L= x.shape
        # else:
        #     B,L,D= data.shape
            
        if self.data_type == "phoneme":
            x = self.embed(x).squeeze(1)
            if x_lens is not None:
                x_padding_mask = make_non_pad_mask(x_lens).to(x.device).unsqueeze(-2)
            else:
                x_padding_mask = None
            x, x_padding_mask = self.encoders(x, x_padding_mask)
        elif self.data_type == "mel":
            x = self.conv(x)
            x = self.embed(x)
            #序列长度下降8倍，相应得mask长度缩减8倍
            if x_lens is not None:
                x_padding_mask = make_non_pad_mask(torch.ceil(x_lens/8)).to(x.device).unsqueeze(-2)
            else:
                x_padding_mask = None
            x = self.encoders(x, x_padding_mask)
        elif self.data_type == "video":
            x = self.avg_pooling(x)
            x = self.projection(x.transpose(1,2)).transpose(1,2)
            x = self.embed(x)
            if x_lens is not None:
                x_padding_mask = make_non_pad_mask(torch.ceil(x_lens/4)).to(x.device).unsqueeze(-2)
            else:
                x_padding_mask = None
            x = self.encoders(x, x_padding_mask)
        return x