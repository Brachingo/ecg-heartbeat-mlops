import argparse
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix

from config import TrainConfig
from dataset import get_loaders
from model import ECGClassifier


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(
        cfg.data_dir / cfg.train_file,
        cfg.data_dir / cfg.test_file,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
    )

    model = ECGClassifier(num_classes=cfg.num_classes, input_size=cfg.input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "architecture": "1D-CNN",
            "dataset": "MIT-BIH Arrhythmia",
        },
    )
    wandb.watch(model, log="all", log_freq=100)

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)

        scheduler.step()

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        train_acc = train_correct / train_total

        print(f"Epoch {epoch:03d}/{cfg.epochs} | "
              f"train_loss={train_loss/train_total:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss / train_total,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.model_dir / cfg.model_name)
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    # Final evaluation on test set
    model.load_state_dict(torch.load(cfg.model_dir / cfg.model_name, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(classification_report(labels, preds, target_names=["Normal", "Suprav.", "Ventr.", "Fusion", "Unknown"]))

    wandb.log({
        "test/loss": test_loss,
        "test/acc": test_acc,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=labels, preds=preds,
            class_names=["Normal", "Suprav.", "Ventr.", "Fusion", "Unknown"],
        ),
    })

    artifact = wandb.Artifact("ecg-model", type="model")
    artifact.add_file(str(cfg.model_dir / cfg.model_name))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr

    train(cfg)
