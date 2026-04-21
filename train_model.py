"""
HandGlow Sign Language Model Trainer
=====================================
Trains a neural network on the recorded dataset and exports
a JSON model that can be loaded directly in the browser.

Usage:
  python train_model.py --csv handglow_dataset.csv
  python train_model.py --csv handglow_dataset.csv --hand right
  python train_model.py --csv handglow_dataset.csv --hand left
  python train_model.py --csv handglow_dataset.csv --hand both

Options:
  --csv        Path to the exported CSV from sign_recorder.html
  --hand       Which hand data to train on: right, left, both (default: both)
               - 'right': train only on right-hand recordings
               - 'left':  train only on left-hand recordings
               - 'both':  use all data + auto-mirror 'either_hand' signs
  --epochs     Training epochs (default: 100)
  --lr         Learning rate (default: 0.001)
  --out        Output directory for the model (default: ./tfjs_model)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# ── Feature columns ──
FLEX_RIGHT = ['R.Thumb', 'R.Index', 'R.Middle', 'R.Ring', 'R.Pinky']
FLEX_LEFT  = ['L.Thumb', 'L.Index', 'L.Middle', 'L.Ring', 'L.Pinky']
IMU_RIGHT  = ['aX1', 'aY1', 'aZ1', 'gX1', 'gY1', 'gZ1']
IMU_LEFT   = ['aX2', 'aY2', 'aZ2', 'gX2', 'gY2', 'gZ2']
ALL_FEATURES = FLEX_RIGHT + FLEX_LEFT + IMU_RIGHT + IMU_LEFT  # 22 features


def mirror_row(row):
    """Swap left↔right channels to create a mirrored version of a sample."""
    mirrored = row.copy()
    # Swap flex: R↔L
    for r, l in zip(FLEX_RIGHT, FLEX_LEFT):
        mirrored[r], mirrored[l] = row[l], row[r]
    # Swap IMU: MPU1↔MPU2
    for r, l in zip(IMU_RIGHT, IMU_LEFT):
        mirrored[r], mirrored[l] = row[l], row[r]
    # Swap recorded_hand tag
    if mirrored.get('recorded_hand') == 'right':
        mirrored['recorded_hand'] = 'left'
    elif mirrored.get('recorded_hand') == 'left':
        mirrored['recorded_hand'] = 'right'
    return mirrored


def load_and_prepare(csv_path, hand_mode):
    """Load CSV, filter by hand, augment either_hand signs."""
    print(f"\n[LOAD] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Raw samples: {len(df)}")
    print(f"   Signs found: {df['label'].nunique()} → {sorted(df['label'].unique())}")

    # Check for hand_mode and recorded_hand columns
    has_hand_mode = 'hand_mode' in df.columns
    has_rec_hand  = 'recorded_hand' in df.columns

    if has_hand_mode:
        print(f"   Hand modes: {dict(df['hand_mode'].value_counts())}")
    if has_rec_hand:
        print(f"   Recorded hands: {dict(df['recorded_hand'].value_counts())}")

    # ── Filter by --hand argument ──
    if hand_mode == 'right' and has_rec_hand:
        df = df[df['recorded_hand'] == 'right']
        print(f"   → Filtered to RIGHT hand only: {len(df)} samples")

    elif hand_mode == 'left' and has_rec_hand:
        df = df[df['recorded_hand'] == 'left']
        print(f"   → Filtered to LEFT hand only: {len(df)} samples")

    elif hand_mode == 'both':
        # Auto-mirror: only for signs that were recorded with ONE hand only
        if has_hand_mode and has_rec_hand:
            either_df = df[df['hand_mode'] == 'either_hand']
            if len(either_df) > 0:
                # Check per sign: only mirror if missing the other hand
                to_mirror = []
                for sign in either_df['label'].unique():
                    sign_df = either_df[either_df['label'] == sign]
                    hands_recorded = set(sign_df['recorded_hand'].unique())
                    if hands_recorded == {'right'}:
                        to_mirror.append(sign_df)
                        print(f"   -> {sign}: only RIGHT recorded, auto-mirroring to LEFT")
                    elif hands_recorded == {'left'}:
                        to_mirror.append(sign_df)
                        print(f"   -> {sign}: only LEFT recorded, auto-mirroring to RIGHT")
                    else:
                        print(f"   -> {sign}: both hands recorded, no mirroring needed")
                if to_mirror:
                    mirror_src = pd.concat(to_mirror, ignore_index=True)
                    mirrored_rows = [mirror_row(row) for _, row in mirror_src.iterrows()]
                    df = pd.concat([df, pd.DataFrame(mirrored_rows)], ignore_index=True)
                    print(f"   -> After mirroring: {len(df)} samples")
        print(f"   -> Using ALL data: {len(df)} samples")

    # ── Extract features and labels ──
    active_features = ALL_FEATURES
    if hand_mode == 'right':
        active_features = FLEX_RIGHT + IMU_RIGHT
        print(f"   -> Subsetting to RIGHT features only ({len(active_features)} features)")
    elif hand_mode == 'left':
        active_features = FLEX_LEFT + IMU_LEFT
        print(f"   -> Subsetting to LEFT features only ({len(active_features)} features)")

    # Ensure all active features exist
    missing = [c for c in active_features if c not in df.columns]
    if missing:
        print(f"   [WARN] Missing columns: {missing}")
        sys.exit(1)

    X = df[active_features].values.astype(np.float32)
    y = df['label'].values

    print(f"\n[DATA] Final dataset: {X.shape[0]} samples × {X.shape[1]} features")
    for label in sorted(set(y)):
        count = np.sum(y == label)
        print(f"   {label:>15}: {count} samples")

    return X, y, active_features


class SignNet(nn.Module):
    """Simple but effective MLP for sign classification."""
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train(X, y, active_features, epochs, lr, out_dir):
    """Train the model and export for browser use."""

    # ── Encode labels ──
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = list(le.classes_)
    n_classes = len(labels)
    print(f"\n[TAG]  Labels: {labels}")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Normalize ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # ── Convert to tensors ──
    X_train_t = torch.from_numpy(X_train_scaled)
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t  = torch.from_numpy(X_test_scaled)
    y_test_t  = torch.from_numpy(y_test).long()

    # ── Build model ──
    model = SignNet(X.shape[1], n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # ── Training loop ──
    print(f"\n[GO] Training for {epochs} epochs...")
    best_acc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Evaluate
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).argmax(dim=1)
                acc = (preds == y_test_t).float().mean().item()
                train_preds = model(X_train_t).argmax(dim=1)
                train_acc = (train_preds == y_train_t).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"   Epoch {epoch:>4d} | Loss: {loss.item():.4f} | "
                  f"Train: {train_acc*100:.1f}% | Test: {acc*100:.1f}%")

    # ── Load best model ──
    if best_state:
        model.load_state_dict(best_state)
    print(f"\n[OK] Best test accuracy: {best_acc*100:.1f}%")

    # ── Final evaluation ──
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()

    print(f"\n[REPORT] Classification Report:")
    print(classification_report(y_test, preds, target_names=labels))

    print(f"[DATA] Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    # Print header
    header = "        " + "  ".join(f"{l:>6}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {labels[i]:>6}" + "  ".join(f"{v:>6}" for v in row))

    # ── Export for browser ──
    os.makedirs(out_dir, exist_ok=True)
    export_for_browser(model, scaler, active_features, labels, best_acc, out_dir)


def export_for_browser(model, scaler, active_features, labels, accuracy, out_dir):
    """
    Export model weights as JSON so the browser can run inference
    without needing TensorFlow.js. Pure JavaScript forward pass.
    """
    print(f"\n[EXPORT] Exporting model to {out_dir}/...")

    # Extract all weights and biases as lists
    layers = []
    state = model.state_dict()

    # Map PyTorch layers to a simple JSON format
    # Our model: Linear→BN→ReLU→Drop → Linear→BN→ReLU→Drop → Linear→ReLU → Linear
    layer_map = [
        # (weight_key, bias_key, bn_weight, bn_bias, bn_mean, bn_var, activation)
        ('net.0', 'net.1', 'relu'),   # Linear(22,128) + BN + ReLU
        ('net.4', 'net.5', 'relu'),   # Linear(128,64) + BN + ReLU
        ('net.8', None, 'relu'),      # Linear(64,32) + ReLU
        ('net.10', None, 'softmax'),  # Linear(32,n_classes) + Softmax
    ]

    for linear_prefix, bn_prefix, activation in layer_map:
        layer_info = {
            'weights': state[f'{linear_prefix}.weight'].numpy().T.tolist(),  # transpose for JS
            'bias': state[f'{linear_prefix}.bias'].numpy().tolist(),
            'activation': activation,
        }
        if bn_prefix:
            layer_info['bn'] = {
                'gamma': state[f'{bn_prefix}.weight'].numpy().tolist(),
                'beta': state[f'{bn_prefix}.bias'].numpy().tolist(),
                'mean': state[f'{bn_prefix}.running_mean'].numpy().tolist(),
                'var': state[f'{bn_prefix}.running_var'].numpy().tolist(),
                'eps': 1e-5,
            }
        layers.append(layer_info)

    model_json = {
        'format': 'handglow_mlp_v1',
        'created': datetime.now().isoformat(),
        'accuracy': round(accuracy, 4),
        'labels': labels,
        'n_features': len(scaler.mean_),
        'feature_names': active_features,
        'normalization': {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
        },
        'layers': layers,
    }

    model_path = os.path.join(out_dir, 'model.json')
    with open(model_path, 'w') as f:
        json.dump(model_json, f)

    size_kb = os.path.getsize(model_path) / 1024
    print(f"   [OK] {model_path} ({size_kb:.1f} KB)")
    print(f"   [DATA] Accuracy: {accuracy*100:.1f}%")
    print(f"   [TAG]  Labels: {labels}")
    print(f"\n[DONE] Done! Load this model in live_sign_reader.html")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HandGlow Model Trainer')
    parser.add_argument('--csv', required=True, help='Path to dataset CSV')
    parser.add_argument('--hand', default='both', choices=['right', 'left', 'both'],
                        help='Hand mode: right, left, or both (default: both)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--out', default='./tfjs_model', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERR] File not found: {args.csv}")
        sys.exit(1)

    X, y, active_features = load_and_prepare(args.csv, args.hand)
    train(X, y, active_features, args.epochs, args.lr, args.out)
