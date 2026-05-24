#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:40:03 2026

@author: muddy
"""

"""
train_classifier.py  —  Classify cloud shadows from 4 folders
================================================================

Expects this folder structure:
    demo_bakes/
        cumulus_low_tau/        bake_0000.jpg, bake_0001.jpg, ...
        cumulus_high_tau/       bake_0000.jpg, bake_0001.jpg, ...
        stratocumulus_low_tau/  bake_0000.jpg, bake_0001.jpg, ...
        stratocumulus_high_tau/ bake_0000.jpg, bake_0001.jpg, ...

Usage:
    python train_classifier.py --data_dir /path/to/demo_bakes
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

IMG_SIZE = 30   # resize all images to this


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_dataset(data_dir, img_size=IMG_SIZE):
    data_dir = Path(data_dir)

    # each subfolder is a class
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    print(f"  Found {len(class_names)} classes: {class_names}")

    images = []
    labels = []

    for class_id, class_dir in enumerate(class_dirs):
        jpgs = sorted(class_dir.glob("*.jpg"))
        print(f"  {class_dir.name}: {len(jpgs)} images")

        for jpg in jpgs:
            img = Image.open(jpg).convert("L").resize((img_size, img_size))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(class_id)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"  Total: {len(images)} images, shape: {images.shape}")
    return images, labels, class_names


def train_test_split(images, labels, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = len(images)
    classes = np.unique(labels)

    train_idx, test_idx = [], []
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_test = max(1, int(len(c_idx) * test_frac))
        test_idx.extend(c_idx[:n_test])
        train_idx.extend(c_idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    return (images[train_idx], labels[train_idx],
            images[test_idx], labels[test_idx])


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (for sklearn)
# ═══════════════════════════════════════════════════════════════════════

def extract_features(images):
    features = []
    for img in images:
        f = []
        h, w = img.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4

        # intensity stats
        f.append(img.mean())
        f.append(img.std())
        f.append(np.median(img))
        f.append(np.percentile(img, 10))
        f.append(np.percentile(img, 90))
        f.append(img.min())
        f.append(img.max())
        f.append((img < 0.5).mean())
        f.append((img < 0.1).mean())
        f.append((img > 0.9).mean())

        # spatial
        center = img[cy-r:cy+r, cx-r:cx+r]
        edge_mask = np.ones_like(img, dtype=bool)
        edge_mask[cy-r:cy+r, cx-r:cx+r] = False
        f.append(center.mean())
        f.append(img[edge_mask].mean())
        f.append(center.mean() - img[edge_mask].mean())

        row_means = img.mean(axis=1)
        col_means = img.mean(axis=0)
        f.append(row_means.std())
        f.append(col_means.std())
        f.append(col_means.std() / (row_means.std() + 1e-8))

        shadow_mask = img < 0.7
        f.append(shadow_mask.mean())

        # gradients
        gy = np.diff(img, axis=0)
        gx = np.diff(img, axis=1)
        f.append(np.mean(gy**2))
        f.append(np.mean(gx**2))
        f.append(np.mean(gy**2) + np.mean(gx**2))

        # frequency
        fft = np.fft.fft2(img)
        fft_mag = np.abs(fft)
        fft_mag[0, 0] = 0
        low_r = min(h, w) // 8
        low_mask = np.zeros_like(fft_mag, dtype=bool)
        low_mask[:low_r, :low_r] = True
        low_mask[:low_r, -low_r:] = True
        low_mask[-low_r:, :low_r] = True
        low_mask[-low_r:, -low_r:] = True
        total_energy = fft_mag.sum() + 1e-8
        f.append(fft_mag[low_mask].sum() / total_energy)
        f.append(fft_mag[~low_mask].sum() / total_energy)

        # radial profile
        yy, xx = np.ogrid[:h, :w]
        r_map = np.sqrt((yy - cy)**2 + (xx - cx)**2).astype(int)
        max_r = min(cy, cx)
        radial = np.zeros(max_r)
        for ri in range(max_r):
            mask = r_map == ri
            if mask.any():
                radial[ri] = img[mask].mean()
        indices = np.linspace(0, max_r - 1, 5).astype(int)
        for idx in indices:
            f.append(radial[idx])

        features.append(f)

    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
# SKLEARN CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════

def train_sklearn(X_train, y_train, X_test, y_test, class_names):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("\n  Extracting features...")
    F_train = extract_features(X_train)
    F_test = extract_features(X_test)
    print(f"  {F_train.shape[1]} features per image")

    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)
    F_test_s = scaler.transform(F_test)

    # Random Forest
    print("\n" + "─"*50)
    print("  RANDOM FOREST")
    print("─"*50)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 random_state=42, n_jobs=-1)
    rf.fit(F_train_s, y_train)
    pred_rf = rf.predict(F_test_s)

    acc = accuracy_score(y_test, pred_rf)
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, pred_rf, target_names=class_names)}")
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, pred_rf))

    # MLP
    print("\n" + "─"*50)
    print("  MLP (128, 64)")
    print("─"*50)

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                         random_state=42, early_stopping=True,
                         validation_fraction=0.15)
    mlp.fit(F_train_s, y_train)
    pred_mlp = mlp.predict(F_test_s)

    acc = accuracy_score(y_test, pred_mlp)
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, pred_mlp, target_names=class_names)}")
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, pred_mlp))

    return rf, mlp, scaler


# ═══════════════════════════════════════════════════════════════════════
# PYTORCH CNN
# ═══════════════════════════════════════════════════════════════════════

def train_pytorch(X_train, y_train, X_test, y_test, class_names,
                  epochs=50, batch_size=32, lr=1e-3):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("\n  PyTorch not installed — skipping CNN.")
        print("  Install with: pip install torch")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    n_classes = len(class_names)

    X_tr = torch.from_numpy(X_train[:, None, :, :])
    X_te = torch.from_numpy(X_test[:, None, :, :])
    y_tr = torch.from_numpy(y_train).long()
    y_te = torch.from_numpy(y_test).long()

    train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(X_te, y_te),
                         batch_size=batch_size)

    class ShadowCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = ShadowCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n  Training CNN for {epochs} epochs...")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc':>8}")
    print("  " + "─" * 26)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # evaluate
        model.eval()
        all_pred = []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                pred = model(xb)
                all_pred.append(pred.cpu().argmax(dim=1))

        pred_all = torch.cat(all_pred).numpy()
        acc = (pred_all == y_test).mean()
        scheduler.step(total_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  {epoch+1:5d}  {total_loss:8.3f}  {acc:8.4f}")

    # final report
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n  Final accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, pred_all, target_names=class_names)}")
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, pred_all))

    # save
    torch.save(model.state_dict(), "shadow_classifier.pth")
    print("  Model saved to shadow_classifier.pth")

    return model


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Classify cloud shadow images")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to folder containing class subfolders")
    parser.add_argument("--skip_pytorch", action="store_true",
                        help="Only run sklearn, skip PyTorch CNN")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print("="*50)
    print("  Cloud Shadow Classifier")
    print("="*50)

    # load
    print("\n  Loading images...")
    images, labels, class_names = load_dataset(args.data_dir)

    # split
    print("\n  Splitting train/test...")
    X_train, y_train, X_test, y_test = train_test_split(images, labels)

    # sklearn
    print("\n" + "="*50)
    print("  SKLEARN CLASSIFIERS")
    print("="*50)
    rf, mlp, scaler = train_sklearn(X_train, y_train, X_test, y_test, class_names)

    # pytorch
    if not args.skip_pytorch:
        print("\n" + "="*50)
        print("  PYTORCH CNN")
        print("="*50)
        model = train_pytorch(X_train, y_train, X_test, y_test, class_names,
                              epochs=args.epochs)

    print("\n  Done.")


if __name__ == "__main__":
    main()