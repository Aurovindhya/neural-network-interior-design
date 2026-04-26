"""
train.py

Training script for the interior style classifier.

Two-phase training:
  Phase 1 (--freeze-epochs): Backbone frozen, only classifier head trains
  Phase 2 (remaining epochs): Top 3 backbone blocks unfrozen, fine-tuned at lower LR

Usage:
    python model/train.py
    python model/train.py --epochs 20 --batch-size 16 --data-dir data/images
    python model/train.py --epochs 20 --output weights/best_model.pth --no-phase2
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.model import InteriorStyleClassifier, STYLE_CLASSES
from model.dataset import build_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = InteriorStyleClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Phase 1: frozen backbone ---
    print(f"\n=== Phase 1: Training classifier head (epochs 1–{args.freeze_epochs}) ===")
    model.freeze_backbone()
    print(f"Trainable params: {model.trainable_params():,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.freeze_epochs)

    best_val_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.freeze_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{args.freeze_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "phase": 1, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, optimizer, epoch, val_acc, class_names, output_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    # --- Phase 2: fine-tune top layers ---
    if not args.no_phase2 and args.epochs > args.freeze_epochs:
        remaining = args.epochs - args.freeze_epochs
        print(f"\n=== Phase 2: Fine-tuning top layers (epochs {args.freeze_epochs + 1}–{args.epochs}) ===")

        model.unfreeze_top_layers(num_blocks=3)
        print(f"Trainable params: {model.trainable_params():,}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 0.1,  # lower LR for fine-tuning
            weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

        for epoch in range(args.freeze_epochs + 1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"{elapsed:.1f}s"
            )
            history.append({"epoch": epoch, "phase": 2, "train_loss": train_loss,
                            "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _save_checkpoint(model, optimizer, epoch, val_acc, class_names, output_path)
                print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Weights saved to: {output_path}")
    return history


def _save_checkpoint(model, optimizer, epoch, val_acc, class_names, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": class_names,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train interior style classifier")
    parser.add_argument("--data-dir", default="data/images")
    parser.add_argument("--output", default="weights/best_model.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs to train with frozen backbone (Phase 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-phase2", action="store_true",
                        help="Skip Phase 2 fine-tuning")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
