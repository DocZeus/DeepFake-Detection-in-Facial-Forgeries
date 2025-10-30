# deepfake_hodff/models.py
from typing import Tuple
import torch
import torch.nn as nn
import timm
from facenet_pytorch import InceptionResnetV1

from deepfake_hodff.config import *

class InceptionResnetV1Head(nn.Module):
    """
    Pretrained FaceNet (InceptionResnetV1) -> 512-d embed -> Dense(128)->Dense(68)
    """
    def __init__(self, out_dim=FUSE_DIM_V1, dropout=0.3):
        super().__init__()
        self.base = InceptionResnetV1(pretrained='vggface2', classify=False)  # 512 embedding
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # x01: (B,3,H,W) in [0,1] -> normalize to [-1,1]
        x = (x01 - 0.5) / 0.5
        emb = self.base(x)  # (B,512)
        return self.head(emb)  # (B,out_dim)

class InceptionResnetV2Head(nn.Module):
    """
    timm inception_resnet_v2 (ImageNet) -> GAP -> 1536 -> Dense(128)->Dense(68)
    """
    def __init__(self, out_dim=FUSE_DIM_V2, dropout=0.3):
        super().__init__()
        self.base = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=0, global_pool='avg') # (B,1536)
        self.head = nn.Sequential(
            nn.Linear(1536, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )
        # timm normalizer
        cfg = timm.data.resolve_model_data_config(self.base)
        self.mean = torch.tensor(cfg['mean']).view(1,3,1,1)
        self.std  = torch.tensor(cfg['std']).view(1,3,1,1)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # normalize with ImageNet stats
        if x01.device != self.mean.device:
            self.mean = self.mean.to(x01.device)
            self.std = self.std.to(x01.device)
        x = (x01 - self.mean) / self.std
        feat = self.base(x)   # (B,1536)
        return self.head(feat)

class DualBackboneFusion(nn.Module):
    """
    TimeDistributed fusion of V1 and V2 heads -> concat -> (B,T,FUSED_DIM)
    """
    def __init__(self):
        super().__init__()
        self.v1 = InceptionResnetV1Head()
        self.v2 = InceptionResnetV2Head()

    def forward(self, x01_btchw: torch.Tensor) -> torch.Tensor:
        # x01_btchw: (B,T,3,H,W) in [0,1]
        B,T,C,H,W = x01_btchw.shape
        x = x01_btchw.reshape(B*T, C, H, W)
        f1 = self.v1(x)            # (B*T, 68)
        f2 = self.v2(x)            # (B*T, 68)
        f = torch.cat([f1,f2], dim=1)   # (B*T, 136)
        f = f.view(B, T, -1)       # (B,T,136)
        return f

class BiLSTMClassifier(nn.Module):
    """
    BiLSTM(128) -> BiLSTM(64) -> Dense(ReLU) -> Dense(2).
    Frame-level logits: (B,T,2). (Sec. 4.4)
    """
    def __init__(self, in_dim=FUSED_DIM, h1=BILSTM_H1, h2=BILSTM_H2, p=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=h1, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=2*h1, hidden_size=h2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p)
        self.fc = nn.Sequential(
            nn.Linear(2*h2, 64), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(64, 2)        # softmax at inference
        )

    def forward(self, f_btf: torch.Tensor) -> torch.Tensor:
        o1,_ = self.lstm1(f_btf)        # (B,T,2*h1)
        o2,_ = self.lstm2(o1)           # (B,T,2*h2)
        o2 = self.dropout(o2)
        logits = self.fc(o2)            # (B,T,2)
        return logits

class HODFF_DD(nn.Module):
    """
    Full model: DualBackboneFusion + BiLSTMClassifier
    """
    def __init__(self):
        super().__init__()
        self.backbones = DualBackboneFusion()
        self.temporal = BiLSTMClassifier()

    def forward(self, x01_btchw: torch.Tensor) -> torch.Tensor:
        f = self.backbones(x01_btchw)   # (B,T,136)
        logits = self.temporal(f)       # (B,T,2)
        return logits
