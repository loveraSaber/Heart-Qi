import torch.nn as nn
class ConvBlock(nn.Module):
    """多尺度时间维度降维"""
    
    def __init__(self, mel_bins=80, reduction_factors=[2, 2, 2]):
        super().__init__()
        self.mel_bins = mel_bins
        self.reduction_factors = reduction_factors
        
        # 为每个降维倍数创建卷积层
        self.stages = nn.ModuleList()
        for factor in reduction_factors:
            stage = nn.Sequential(
                nn.Conv1d(
                    mel_bins, mel_bins,
                    kernel_size=factor * 2 - 1,
                    stride=factor,
                    padding=factor - 1,
                    groups=mel_bins,
                    bias=False
                ),
                nn.BatchNorm1d(mel_bins),
                nn.ReLU()
            )
            # self._init_conv_weight(conv, factor)
            self.stages.append(stage)
    
    # def _init_conv_weight(self, conv, factor):
    #     """初始化卷积权重"""
    #     kernel_size = factor * 2 - 1
    #     weight = torch.ones((self.mel_bins, 1, kernel_size)) / kernel_size
    #     with torch.no_grad():
    #         conv.weight.data = weight
    
    def forward(self, x, reduction_index=0):
        """选择特定降维倍数的卷积"""
        x = x.transpose(1,2)
        for stage in self.stages:
            x = stage(x)
        return x.transpose(1,2)