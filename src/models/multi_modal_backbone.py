import torch
import torch.nn as nn
from models.encoders.encoder import Encoder
from models.backend.label_prediction import LabelPrediction
from models.backend.predict_layer import Predict_Layer

class E2E(nn.Module):
    def __init__(self, cfg, scale_type):
        super(E2E, self).__init__()
        #声明mel处理
        if cfg.mel.label == True:
            self.mel_encoder = Encoder(cfg.mel, "mel")
            self.label_prediction_mel = LabelPrediction(cfg, scale_type, "mel")
        #声明视频处理
        if cfg.video.label == True:
            self.video_encoder = Encoder(cfg.video, "video")
            self.label_prediction_video = LabelPrediction(cfg, scale_type, "video")
        #声明phoneme处理
        if cfg.phoneme.label == True:
            self.phoneme_encoder = Encoder(cfg.phoneme, "phoneme")
            self.label_prediction_phoneme = LabelPrediction(cfg, scale_type, "phoneme")
        

        self.predict = Predict_Layer(scale_type=scale_type,  num_modalities = cfg["modal_num"])

        # self.label_fusion = LabelFusion(config, device) 
        # self.label_caulate = LabelCalculate(config)
        
        # self.loss_function = Loss_Function(config, device)
        #self.embedding = nn.Embedding(config["phoneme"]["vocab_size"], 128, padding_idx=0)

    def forward(self, mel, mel_lens, video, video_lens, phoneme, phoneme_lens, label, label_score ):
        #执行mel
        result = {}
        if hasattr(self, "mel_encoder"): #检测模块是否存在
            mel_encoder_output = self.mel_encoder(mel,mel_lens)
            prediction_mel = self.label_prediction_mel(mel_encoder_output)
            result["mel"] = prediction_mel
        #执行video
        if hasattr(self, "video_encoder"):
            video_encoder_output = self.video_encoder(video,video_lens)
            prediction_video = self.label_prediction_video(video_encoder_output)
            result["video"] = prediction_video
        #执行phoneme
        if hasattr(self, "phoneme_encoder"):
            phoneme_encoder_output = self.phoneme_encoder(phoneme,phoneme_lens)
            prediction_phoneme = self.label_prediction_phoneme(phoneme_encoder_output)
            result["phoneme"] = prediction_phoneme

        loss = self.predict(result, label_score)


        return loss
    
    @torch.no_grad()
    def inference(self, batch):
        mel, mel_lens, mel_mask, video, video_lens, phoneme, phoneme_lens, label, label_score = batch

        mel_encoder_output = self.mel_encoder(mel,mel_mask)
        video_encoder_output = self.video_encoder(video,video_mask)
        phoneme_encoder_output = self.phoneme_encoder(phoneme,phoneme_mask)
        
        prediction_mel = self.label_prediction_mel(mel_encoder_output)
        prediction_video = self.label_prediction_video(video_encoder_output)
        prediction_phoneme = self.label_prediction_phoneme(phoneme_encoder_output)
        
        label_score_prediction = self.label_fusion(prediction_mel, prediction_video, prediction_phoneme)
        label_caculate_text, label_caculate_num = self.label_caulate(label_score_prediction)

        

        return label_score_prediction, label_caculate_text

        
        


        