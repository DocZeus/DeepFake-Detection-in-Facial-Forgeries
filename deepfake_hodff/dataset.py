# deepfake_hodff/dataset.py
from pathlib import Path
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from deepfake_hodff.config import *

class FaceSeqDataset(Dataset):
    def __init__(self, index_csv: Path, split: str, seq_len=SEQ_LEN, stride=SEQ_STRIDE, augment=False):
        self.df = pd.read_csv(index_csv)
        self.df = self.df[self.df["split"]==split]
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment and (split=="train")

        # group frames by video
        self.groups = (self.df.groupby(["video_id","label"])["image_path"]
                            .apply(list).reset_index())
        # pre-generate sequence start indices per video
        self.samples = []
        for _,r in self.groups.iterrows():
            paths = sorted(r["image_path"])
            if len(paths)==0: continue
            # sliding windows
            for start in range(0, max(1, len(paths)-seq_len+1), stride):
                self.samples.append((r["video_id"], r["label"], start))
            # ensure at least one sample
            if len(paths) < seq_len:
                self.samples.append((r["video_id"], r["label"], 0))

        self.aug = A.Compose([
            A.HorizontalFlip(p=AUG_HFLIP_P),
            A.Rotate(limit=AUG_ROT_DEG, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=AUG_BRIGHTNESS,
                                       contrast_limit=AUG_CONTRAST, p=0.5),
        ])

    def __len__(self):
        return len(self.samples)

    def _read_img(self, p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment:
            img = self.aug(image=img)["image"]
        img = img.astype(np.float32) / 255.0
        return img

    def __getitem__(self, i):
        vid, label, start = self.samples[i]
        paths = sorted(self.df[self.df["video_id"]==vid]["image_path"].tolist())
        seq = []
        for k in range(self.seq_len):
            idx = min(start+k, len(paths)-1)
            seq.append(self._read_img(paths[idx]))
        arr = np.stack(seq, axis=0) # T,H,W,C
        arr = np.transpose(arr, (0,3,1,2)) # T,C,H,W
        x = torch.from_numpy(arr)          # float32, in [0,1]
        y = 1 if label=="fake" else 0
        y = torch.tensor(y, dtype=torch.long)
        return x, y, vid, paths[start:min(start+self.seq_len,len(paths))]
