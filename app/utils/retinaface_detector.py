import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from collections import deque
from datetime import datetime
from batch_face import RetinaFace
from tqdm import tqdm
import gc
from PIL import Image
import torch


class Stability:
    def __init__(self, alpha=0.5, window_size=7):
        self.alpha = alpha
        self.window_size = window_size
        self.prev_ema = None
        self.history = deque(maxlen=window_size)

    def smooth(self, landmarks, score=None):
        if landmarks is None:
            return None
        if self.prev_ema is None:
            self.prev_ema = landmarks.copy()
            self.history.append(landmarks.copy())
            return landmarks
        alpha = self.alpha
        if score is not None:
            alpha = min(0.95, max(0.5, score))
        smoothed = alpha * landmarks + (1 - alpha) * self.prev_ema
        self.prev_ema = smoothed.copy()
        self.history.append(smoothed)
        return np.mean(self.history, axis=0)


class RetinaFace_LandmarksDetector:  
    def __init__(self, threshold=0.95, max_size=640):
        self.detector = RetinaFace(gpu_id=0)
        self.threshold = threshold
        self.max_size = max_size

    def detect(self, batch_images):
        if not batch_images:
            return []
        all_faces = self.detector(
            batch_images,
            threshold=self.threshold,
            max_size=self.max_size,
            batch_size=len(batch_images)
        )
        all_results = []
        for faces in all_faces:
            if not faces:
                all_results.append(None)
                continue
            face = max(faces, key=lambda f: (f[0][2]-f[0][0]) * (f[0][3]-f[0][1]))
            box, kps, score = face[0], face[1], face[2]
            all_results.append((box.astype(int), kps.astype(float), score))
        return all_results

    def __call__(self, frames):
        return self.detect(frames)
    
def align_face(frame, landmarks, output_size=(224, 224)):
    # 112x112 的对齐模板
    ref_points = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    # 缩放到目标尺寸
    ref_points = ref_points * (output_size[0] / 112.0)

    src_points = landmarks.astype(np.float32)

    # 计算仿射矩阵
    M, _ = cv2.estimateAffinePartial2D(src_points, ref_points, method=cv2.LMEDS) #计算仿射变换矩阵
    aligned = cv2.warpAffine(frame, M, output_size, borderValue=0.0) #输入人脸对齐到目标尺度
    return aligned

def process_video(video_path, detector, batch_size=64,  output_path="output_faces.mp4", alpha=0.7, window_size=5, face_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    face_size = face_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, face_size)

    frames_rgb = []
    frames_bgr = []

    stabilizer = Stability(alpha=alpha, window_size=window_size)

    with tqdm(desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_bgr.append(frame)
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(frames_rgb) == batch_size:
                results = detector(frames_rgb)

                for frm, result in zip(frames_bgr, results):
                    if result is None:
                        continue

                    box, kps, score = result
                    smoothed_kps = stabilizer.smooth(kps, score=score)
                    face_aligned = align_face(frm, smoothed_kps, output_size=face_size)
                    writer.write(face_aligned)

                frames_rgb.clear()
                frames_bgr.clear()

            pbar.update(1)

    # 处理剩余帧
    if frames_rgb:
        results = detector(frames_rgb)
        for frm, result in zip(frames_bgr, results):
            if result is None:
                continue
            box, kps, score = result
            smoothed_kps = stabilizer.smooth(kps, score=score)
            face_aligned = align_face(frm, smoothed_kps, output_size=face_size)
            writer.write(face_aligned)

    cap.release()
    writer.release()
    gc.collect()

def get_face(video_path, detector, batch_size=64,  output_path="output_faces.mp4", alpha=0.7, window_size=5, face_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_rgb = []
    frames_bgr = []
    stabilizer = Stability(alpha=alpha, window_size=window_size)
    with tqdm(desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames_rgb) == batch_size:
                results = detector(frames_rgb)
                for frm, result in zip(frames_bgr, results):
                    if result is None:
                        continue
                    box, kps, score = result
                    smoothed_kps = stabilizer.smooth(kps, score=score)
                    face_aligned = align_face(frm, smoothed_kps, output_size=face_size)
                frames_rgb.clear()
                frames_bgr.clear()
            pbar.update(1)

    # 处理剩余帧
    if frames_rgb:
        results = detector(frames_rgb)
        for frm, result in zip(frames_bgr, results):
            if result is None:
                continue
            box, kps, score = result
            smoothed_kps = stabilizer.smooth(kps, score=score)
            face_aligned = align_face(frm, smoothed_kps, output_size=face_size)

    cap.release()
    gc.collect()

    return face_aligned
