# deepfake_hodff/config.py
from pathlib import Path

# Paths
DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT/"raw"/"Celeb-DF-v2"
PROC_ROOT = DATA_ROOT/"processed"
SPLIT_ROOT = DATA_ROOT/"splits"
FEATURE_ROOT = DATA_ROOT/"features"
OUT_ROOT = Path("outputs")
CKPT_DIR = OUT_ROOT/"checkpoints"
LOG_DIR = OUT_ROOT/"logs"

# Splits
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10  # within train; early stopping
TEST_FRACTION = 0.30

# Preprocessing
FPS = 1                  # frame extraction rate
FACE_MARGIN = 20         # pixels around bbox (paper Sec. 4.1)
FACE_SIZE_1 = 224        # intermediate resize
FACE_SIZE_2 = 300        # final input size (paper Sec. 4.1)
MAX_FRAMES_PER_VIDEO = 120  # safety cap

# Sequences
SEQ_LEN = 16
SEQ_STRIDE = 16

# Training
BATCH_SIZE = 10                      # paper Sec. 5.2
LR = 1e-5                            # paper Sec. 5.2
BETAS = (0.9, 0.999)                 # paper Sec. 5.2
EPOCHS = 50
EARLY_STOP_PATIENCE = 4              # paper Sec. 5.2
NUM_WORKERS = 4
DEVICE = "cuda"

# Model
FUSE_DIM_V1 = 68                     # custom dense head units (Sec. 4.2)
FUSE_DIM_V2 = 68                     # custom dense head units (Sec. 4.2)
FUSED_DIM = FUSE_DIM_V1 + FUSE_DIM_V2
BILSTM_H1 = 128                      # (128, then 64) Sec. 4.4
BILSTM_H2 = 64

# Augmentation
AUG_ROT_DEG = 10
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST = 0.15
AUG_HFLIP_P = 0.5
