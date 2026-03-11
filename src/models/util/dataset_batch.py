import os
import json
import numpy as np
from ..util.utils import pad_2D,pad_1D
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, config, drop_last=False, train=True):

        self.train = train
        self.drop_last = drop_last
        self.preprocessed_path = config["path"]["preprocessed_path"]
        self.batch_size = config["optimizer"]["batch_size"]
        
        self.scale_type = config["scale_type"]
        self.mel_dir = os.path.join(self.preprocessed_path, "mel")
        self.video_dir = os.path.join(self.preprocessed_path, "frame")
        self.phoneme_dir = os.path.join(self.preprocessed_path, "phoneme")
        self.label_dir = os.path.join(self.preprocessed_path, "label")
        self.label_score_dir = os.path.join(self.preprocessed_path, "label_score")
        
        if train:
            filename = "train.txt"
        else:
            filename = "val.txt"

        self.basename = self.process_meta(filename)

    def process_meta(self, filename):
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        
        mel_path = os.path.join(self.mel_dir, f"{basename}-mel.npy")
        mel = np.load(mel_path)

        video_path = os.path.join(self.video_dir, f"{basename}-frame.npy")
        video = np.load(video_path)
        
        phoneme_path = os.path.join(self.phoneme_dir, f"{basename}-phoneme.npy")
        phoneme = np.load(phoneme_path)

        label_path = os.path.join(self.label_dir, self.scale_type, f"{basename}.npy")
        label = np.load(label_path)
        
        label_score_path = os.path.join(self.label_score_dir, self.scale_type, f"{basename}.npy")
        label_score = np.load(label_score_path)
        
        sample = {
            "basename": basename,
            "mels": mel,
            "videos": video,
            "phonemes": phoneme,
            "labels": label,
            "labels_score": label_score
        }
        
        return sample

    def reprocess(self, data, idxs):
        basename = [data[idx]["basename"] for idx in idxs]
        
        mels = [data[idx]["mels"] for idx in idxs]
        videos = [data[idx]["videos"] for idx in idxs]
        phonemes = [data[idx]["phonemes"] for idx in idxs]
        labels = [data[idx]["labels"] for idx in idxs]
        labels_score = [data[idx]["labels_score"] for idx in idxs]

        mel_lens = np.array([mel.shape[0] for mel in mels])
        max_mel_len = max(mel_lens)
        
        video_lens = np.array([video.shape[0] for video in videos])
        max_video_len = max(video_lens)
        
        phoneme_lens = np.array([phoneme.shape[0] for phoneme in phonemes])
        max_phoneme_len = max(phoneme_lens)
        
        mels_pad = pad_2D(mels,max_mel_len)
        videos_pad = pad_2D(videos, max_video_len)
        phonemes_pad = pad_1D(phonemes, max_phoneme_len)
    
        if self.train:
            return (
                basename,
                mels_pad,
                mel_lens,
                max_mel_len,
                videos_pad,
                video_lens,
                max_video_len,
                phonemes_pad,
                phoneme_lens,
                max_phoneme_len,
                labels,
                labels_score
            )
            
        else:
            return (
                basename,
                mels_pad,
                mel_lens,
                max_mel_len,
                videos_pad,
                video_lens,
                max_video_len,
                phonemes_pad,
                phoneme_lens,
                max_phoneme_len
            )

    def collate_fn(self, data):
        data_size = len(data)
        idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[:len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()

        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = []
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output
