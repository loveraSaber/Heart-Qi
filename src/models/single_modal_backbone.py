import torch.nn as nn
from .encoders.encoder import Encoder
from .backend.label_prediction import LabelPrediction
from .backend.predict_layer import Predict_Layer
from .classifier import SimpleClassifier

class E2E(nn.Module):
    def __init__(self, cfg, scale_type):
        super(E2E, self).__init__()
        # 声明视频处理
        if cfg.video.label == True:
            self.video_encoder = Encoder(cfg.video, "video")
            self.label_prediction_video = LabelPrediction(cfg, scale_type, "video")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,video, video_lens):
        #执行mel
        result = {}
        #执行video
        if hasattr(self, "video_encoder"):
            video_encoder_output = self.video_encoder(video,video_lens)
            prediction_video = self.label_prediction_video(video_encoder_output)
            result["video"] = prediction_video
        # loss = self.predict(result, label_score)
        # loss = self.criterion(output, target)
        # return loss
    
        
        


        