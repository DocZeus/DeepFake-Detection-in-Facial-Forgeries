# deepfake_hodff/evaluate.py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from deepfake_hodff.config import *
from deepfake_hodff.dataset import FaceSeqDataset
from deepfake_hodff.models import HODFF_DD

@torch.no_grad()
def predict_framewise(model, loader):
    model.eval()
    recs = []
    for x, y, vid, paths in tqdm(loader, desc="predict"):
        x = x.to(DEVICE)
        logits = model(x)  # (B,T,2)
        prob = torch.softmax(logits, dim=-1)[...,1].cpu().numpy()  # (B,T)
        y = y.numpy()
        for b in range(prob.shape[0]):
            for t,pp in enumerate(prob[b]):
                recs.append({"video_id": vid[b], "frame_idx": t, "prob_fake": float(pp), "label": int(y[b])})
    return pd.DataFrame(recs)

def majority_vote(df_pred):
    rows = []
    for vid, g in df_pred.groupby("video_id"):
        y = int(g["label"].iloc[0])
        probs = g["prob_fake"].values
        preds = (probs>=0.5).astype(int)
        maj = int(np.round(preds.mean()))  # 0/1 majority
        rows.append({"video_id": vid, "y_true": y, "y_prob": float(probs.mean()), "y_pred": maj})
    return pd.DataFrame(rows)

def compute_metrics(agg):
    acc = accuracy_score(agg.y_true, agg.y_pred)
    try:
        auc = roc_auc_score(agg.y_true, agg.y_prob)
    except Exception:
        auc = float('nan')
    tn, fp, fn, tp = confusion_matrix(agg.y_true, agg.y_pred).ravel()
    tpr = tp / (tp + fn + 1e-8)  # Recall for positive class
    tnr = tn / (tn + fp + 1e-8)
    prec, rec, f1, _ = precision_recall_fscore_support(agg.y_true, agg.y_pred, average="binary", zero_division=0)
    return dict(Accuracy=acc, AUC=auc, TPR=tpr, TNR=tnr, Precision=prec, Recall=rec, F1=f1)

def main():
    index_csv = SPLIT_ROOT/"index.csv"
    test_ds = FaceSeqDataset(index_csv, "test", augment=False)
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = HODFF_DD().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_DIR/"hodff_dd.pt", map_location=DEVICE))

    df_pred = predict_framewise(model, loader)
    agg = majority_vote(df_pred)

    metr = compute_metrics(agg)
    print(pd.Series(metr))

    # Save reports
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(OUT_ROOT/"frame_predictions.csv", index=False)
    agg.to_csv(OUT_ROOT/"video_predictions.csv", index=False)

if __name__ == "__main__":
    main()
