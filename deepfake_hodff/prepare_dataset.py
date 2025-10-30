# deepfake_hodff/prepare_dataset.py
import argparse, os, random, json
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
from deepfake_hodff.config import *

REAL_DIR_CANDIDATES = ["Real", "real", "Celeb-real", "YouTube-real", "Youtube-real"]
FAKE_DIR_CANDIDATES = ["Fake", "fake", "Celeb-synthesis", "Celeb-syn"]

def _find_label_dirs(root: Path):
    real_dirs = [root/d for d in REAL_DIR_CANDIDATES if (root/d).exists()]
    fake_dirs = [root/d for d in FAKE_DIR_CANDIDATES if (root/d).exists()]
    if not real_dirs or not fake_dirs:
        raise RuntimeError("Could not locate real/fake directories in Celeb-DF-v2 root.")
    return real_dirs, fake_dirs

def _index_videos(root: Path):
    real_dirs, fake_dirs = _find_label_dirs(root)
    rows = []
    for lbl, dirs in [("real", real_dirs), ("fake", fake_dirs)]:
        for d in dirs:
            for p in sorted(d.rglob("*.mp4")):
                rows.append({"video_path": str(p), "label": lbl, "video_id": p.stem})
    df = pd.DataFrame(rows)
    return df

def _train_val_test_split(df: pd.DataFrame, train_frac=TRAIN_FRACTION, val_frac=VAL_FRACTION, seed=1337):
    random.seed(seed)
    vids = df["video_id"].unique().tolist()
    random.shuffle(vids)
    n = len(vids)
    n_train = int(n*train_frac)
    n_val = int(n_train*val_frac)
    train_ids = set(vids[:n_train])
    val_ids = set(list(train_ids)[:n_val])
    test_ids = set(vids[n_train:])
    df["split"] = df["video_id"].apply(lambda x: "train" if x in train_ids else ("val" if x in val_ids else ("test" if x in test_ids else "other")))
    return df

def _ensure_dirs():
    for d in [PROC_ROOT, SPLIT_ROOT]:
        d.mkdir(parents=True, exist_ok=True)

def _resize_keep_inside(x1, y1, x2, y2, w, h, margin):
    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(w, x2 + margin)
    y2m = min(h, y2 + margin)
    return x1m, y1m, x2m, y2m

def _extract_frames_and_faces(df: pd.DataFrame):
    mtcnn = MTCNN(select_largest=True, device=DEVICE if DEVICE=="cuda" else "cpu", post_process=False)
    index_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Videos"):
        vpath = Path(row["video_path"])
        split = row["split"]; label = row["label"]; vid = row["video_id"]
        out_dir = PROC_ROOT/split/label/vid
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps / FPS)))
        idx, saved = 0, 0
        while True:
            ret = cap.grab()
            if not ret: break
            if idx % step == 0:
                ret, frame = cap.retrieve()
                if not ret: break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MTCNN face detection
                boxes, probs = mtcnn.detect(rgb)
                if boxes is not None and len(boxes) > 0:
                    # choose largest
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                    j = int(max(range(len(areas)), key=lambda i: areas[i]))
                    x1, y1, x2, y2 = boxes[j].astype(int)
                    x1, y1, x2, y2 = _resize_keep_inside(x1, y1, x2, y2, w, h, FACE_MARGIN)
                    face = rgb[y1:y2, x1:x2].copy()

                    # resize 224 -> 300 (paper Sec. 4.1)
                    face224 = cv2.resize(face, (FACE_SIZE_1, FACE_SIZE_1), interpolation=cv2.INTER_AREA)
                    face300 = cv2.resize(face224, (FACE_SIZE_2, FACE_SIZE_2), interpolation=cv2.INTER_LINEAR)

                    fname = f"f_{saved:05d}.jpg"
                    cv2.imwrite(str(out_dir/fname), cv2.cvtColor(face300, cv2.COLOR_RGB2BGR))
                    index_rows.append({
                        "split": split, "label": label, "video_id": vid,
                        "frame_idx": saved, "image_path": str(out_dir/fname)
                    })
                    saved += 1
                    if saved >= MAX_FRAMES_PER_VIDEO:
                        break
            idx += 1
        cap.release()
    idx_df = pd.DataFrame(index_rows)
    (SPLIT_ROOT/"index.csv").parent.mkdir(parents=True, exist_ok=True)
    idx_df.to_csv(SPLIT_ROOT/"index.csv", index=False)
    print(f"Indexed frames saved at {SPLIT_ROOT/'index.csv'}")

def main():
    _ensure_dirs()
    df = _index_videos(RAW_ROOT)
    df = _train_val_test_split(df)
    df.to_csv(SPLIT_ROOT/"videos.csv", index=False)
    _extract_frames_and_faces(df)

if __name__ == "__main__":
    main()
