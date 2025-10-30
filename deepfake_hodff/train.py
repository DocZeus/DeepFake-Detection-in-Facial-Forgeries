# deepfake_hodff/train.py
import argparse, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from deepfake_hodff.config import *
from deepfake_hodff.dataset import FaceSeqDataset
from deepfake_hodff.models import HODFF_DD
from deepfake_hodff.optimizer_sho import SpottedHyena

def make_loaders(index_csv):
    train_ds = FaceSeqDataset(index_csv, "train", augment=True)
    val_ds   = FaceSeqDataset(index_csv, "val",   augment=False)
    test_ds  = FaceSeqDataset(index_csv, "test",  augment=False)

    # balance classes in train via weighted sampling
    y = []
    for _,r in train_ds.df.iterrows():
        y.append(1 if r["label"]=="fake" else 0)
    if len(y)==0: y=[0]
    class_sample_count = np.array([ (np.array(y)==t).sum() for t in [0,1] ])
    class_weights = 1./np.maximum(class_sample_count,1)
    sample_weights = np.array([ class_weights[1 if lab=="fake" else 0] for lab in train_ds.df["label"] ])
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, y_true, y_prob = [], [], []
    for x, y, _, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)           # (B,T,3,300,300)
        y = y.to(device)           # (B,)
        optimizer.zero_grad()
        logits = model(x)          # (B,T,2)
        # frame-level CE vs video label: repeat targets across T
        B,T,_ = logits.shape
        loss = criterion(logits.view(B*T,2), y.view(B,1).repeat(1,T).view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # collect probs for video-level rough tracking (avg across frames)
        prob_fake = torch.softmax(logits.detach(), dim=-1)[...,1].mean(dim=1) # (B,)
        y_true.extend(y.cpu().numpy().tolist())
        y_prob.extend(prob_fake.cpu().numpy().tolist())
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    acc = accuracy_score(y_true, [1 if p>=0.5 else 0 for p in y_prob])
    return np.mean(losses), acc, auc

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_prob = [], [], []
    for x, y, _, _ in tqdm(loader, desc="eval", leave=False):
        x = x.to(device); y = y.to(device)
        logits = model(x)
        B,T,_ = logits.shape
        loss = criterion(logits.view(B*T,2), y.view(B,1).repeat(1,T).view(-1))
        losses.append(loss.item())
        prob_fake = torch.softmax(logits, dim=-1)[...,1].mean(dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_prob.extend(prob_fake.cpu().numpy().tolist())
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    acc = accuracy_score(y_true, [1 if p>=0.5 else 0 for p in y_prob])
    return np.mean(losses), acc, auc

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    index_csv = SPLIT_ROOT/"index.csv"
    train_loader, val_loader, test_loader = make_loaders(index_csv)

    model = HODFF_DD().to(DEVICE)

    # Freeze most CNN weights first; train heads+BiLSTM; unfreeze later if desired
    for p in model.backbones.v1.base.parameters(): p.requires_grad = False
    for p in model.backbones.v2.base.parameters(): p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    use_sho = True  # paper optimizer
    if use_sho:
        optimizer = SpottedHyena(params, lr=LR, hyena_steps=1)
    else:
        optimizer = torch.optim.Adam(params, lr=LR, betas=BETAS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, patience = 0.0, 0
    best_path = CKPT_DIR/"hodff_dd.pt"

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_auc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc, va_auc = eval_epoch(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} auc {tr_auc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f} auc {va_auc:.3f}")

        # Early stopping on validation accuracy (patience=4), per Sec. 5.2
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

    # Load best and test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te_loss, te_acc, te_auc = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"Test | loss {te_loss:.4f} acc {te_acc:.3f} auc {te_auc:.3f}")

if __name__ == "__main__":
    main()
