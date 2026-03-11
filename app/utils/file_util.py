from scipy.io import savemat, loadmat
import numpy as np


def save_feature_to_mat(features, mat_path):
    savemat(mat_path, {"video_features": features})

def load_mat(mat_path):
    data = loadmat(mat_path)
    video_features = data['video_features']
    return video_features
