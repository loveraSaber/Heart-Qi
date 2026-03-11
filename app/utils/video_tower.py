import torchvision
torchvision.disable_beta_transforms_warning()
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

class VitTower():
    def __init__(self, model_path, device):
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(model_path, ignore_mismatched_sizes=True)
        self.model = ViTModel.from_pretrained(model_path).to(self.device)
        self.model.eval()  

    def _infer_batch(self, batch_frames):
        if not batch_frames:
            return None

        with torch.inference_mode():
            inputs = self.processor(images=batch_frames, return_tensors="pt")
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

        # 手动断引用（关键）
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return feats
    
    def forward(self, video_path, frame_rate=25, batch_size=32):
        try:
            # 读取视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / frame_rate)) if frame_rate > 0 else 1
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 批处理参数 - 可根据模型和GPU调整
            batch_size = batch_size  # 建议值：16, 32, 64
            features = []
            batch_frames = []
            processed_count = 0
            
            expected_frames = (total_frames + frame_interval - 1) // frame_interval
            
            with tqdm(total=expected_frames) as pbar:
                for frame_count in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 按帧间隔采样
                    if frame_count % frame_interval == 0:
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)
                            batch_frames.append(pil_image)
                            processed_count += 1
                            
                            # 批次处理
                            if len(batch_frames) >= batch_size:
                                inputs = self.processor(images=batch_frames, return_tensors="pt")
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                
                                with torch.no_grad():
                                    outputs = self.model(**inputs)
                                
                                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                                features.append(batch_features)
                                
                                batch_frames = []
                                pbar.update(batch_size)
                                
                        except Exception as frame_error:
                            print(f"处理第{frame_count}帧时出错: {frame_error}")
                            continue
                
                # 处理最后一批
                if batch_frames:
                    inputs = self.processor(images=batch_frames, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    features.append(batch_features)
                    pbar.update(len(batch_frames))
            
            cap.release()
            
            # 合并所有批次特征
            if features:
                features = np.vstack(features)
                print(f"成功提取 {features.shape[0]} 帧特征，特征维度: {features.shape[1]}")
            else:
                features = np.array([])
                print("未提取到有效特征")
                
        except Exception as e:
            print(f"视频处理失败: {e}")
            if 'cap' in locals():
                cap.release()
            features = np.array([])
        return features

    def extract_feature(self, face_align, batch_size=32):
        
        features = []
        batch_frames = []
        processed_count = 0
        try:
            for frame in face_align:
                batch_frames.append(Image.fromarray(frame))
                processed_count += 1

                if len(batch_frames) == batch_size:
                    batch_feat = self._infer_batch(batch_frames)
                    if batch_feat is not None:
                        features.append(batch_feat)
                    batch_frames.clear()

            # 处理最后一批
            if batch_frames:
                batch_feat = self._infer_batch(batch_frames)
                if batch_feat is not None:
                    features.append(batch_feat)
                batch_frames.clear()

            if features:
                features = np.vstack(features)
                print(f"成功提取 {features.shape[0]} 帧特征，特征维度: {features.shape[1]}")
            else:
                features = np.empty((0,))
                print("未提取到有效特征")

            return features

        finally:
            # 强制资源回收（关键）
            del batch_frames
            del features
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
