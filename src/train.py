import argparse
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report

from config import ConfigEntrenamiento
from dataset import obtener_cargadores
from model_cnn import ClasificadorCNN
from model_gru import ClasificadorGRU

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def train(cfg: ConfigEntrenamiento, dataset: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.directorio_modelos.mkdir(parents=True, exist_ok=True)

    # Descargar el artifact del dataset desde W&B
    run = wandb.init(project=cfg.wandb_proyecto, entity=cfg.wandb_entidad,
                     name=f"train-{dataset}", job_type="train",
                     config={"epochs": cfg.epocas, "batch_size": cfg.batch_size,
                             "lr": cfg.tasa_aprendizaje, "dataset": dataset})

    artifact = run.use_artifact(f"{dataset}:latest")
    data_dir = Path(artifact.download())

    train_loader, val_loader, test_loader = obtener_cargadores(
        data_dir / "train.csv",
        data_dir / "test.csv",
        batch_size=cfg.batch_size,
        proporcion_validacion=cfg.proporcion_validacion,
    )

    model = ClasificadorCNN(num_clases=cfg.num_clases, tamano_entrada=cfg.tamano_entrada).to(device) if cfg.model == "CNN" else ClasificadorGRU(num_clases=cfg.num_clases, tamano_entrada=cfg.tamano_entrada).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.tasa_aprendizaje, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epocas)

    best_val_acc = 0.0

    for epoch in range(1, cfg.epocas + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        scheduler.step()

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        train_acc = correct / total

        print(f"Época {epoch:03d}/{cfg.epocas} | "
              f"train_loss={train_loss/total:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        wandb.log({"epoch": epoch,
                   "train/loss": train_loss / total, "train/acc": train_acc,
                   "val/loss": val_loss, "val/acc": val_acc,
                   "lr": scheduler.get_last_lr()[0]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.directorio_modelos / cfg.nombre_modelo)
            print(f"  → Modelo guardado (val_acc={val_acc:.4f})")

    # Evaluación final en test
    model.load_state_dict(torch.load(cfg.directorio_modelos / cfg.nombre_modelo, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"\nTest — loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(classification_report(labels, preds,
          target_names=["Normal", "Supraventricular", "Ventricular", "Fusion", "Desconocido"]))

    wandb.log({"test/loss": test_loss, "test/acc": test_acc,
               "test/confusion_matrix": wandb.plot.confusion_matrix(
                   probs=None, y_true=labels, preds=preds,
                   class_names=["Normal", "Supraventricular", "Ventricular", "Fusion", "Desconocido"])})

    # Subir modelo como artifact
    model_artifact = wandb.Artifact(f"ecg-model-{dataset}", type="model")
    model_artifact.add_file(str(cfg.directorio_modelos / cfg.nombre_modelo))
    run.log_artifact(model_artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mitbih-raw",
                        choices=["mitbih-raw", "mitbih-balanced"], help="Dataset a usar (raw o balanced)")
    parser.add_argument("--model", type=str, default="GRU", choices=["CNN", "GRU"], help="Arquitectura del modelo (CNN o GRU)")
    parser.add_argument("--epocas", type=int, default=None, help="Cantidad de épocas para entrenar")
    parser.add_argument("--batch_size", type=int, default=None, help="Tamaño del batch para entrenamiento")
    parser.add_argument("--lr", type=float, default=None, help="Tasa de aprendizaje para el optimizador")
    args = parser.parse_args()

    cfg = ConfigEntrenamiento()
    cfg.model = args.model
    if args.epocas:     cfg.epocas = args.epocas
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.tasa_aprendizaje = args.lr

    train(cfg, args.dataset)
