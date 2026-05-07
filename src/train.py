import argparse

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report

from config import ConfigEntrenamiento
from dataset import obtener_cargadores
from model import ClasificadorECG


def evaluar(modelo, cargador, criterio, dispositivo):
    modelo.eval()
    perdida_total, correctos, total = 0.0, 0, 0
    todas_predicciones, todas_etiquetas = [], []
    with torch.no_grad():
        for X, y in cargador:
            X, y = X.to(dispositivo), y.to(dispositivo)
            logits = modelo(X)
            perdida = criterio(logits, y)
            perdida_total += perdida.item() * len(y)
            predicciones = logits.argmax(dim=1)
            correctos += (predicciones == y).sum().item()
            total += len(y)
            todas_predicciones.extend(predicciones.cpu().tolist())
            todas_etiquetas.extend(y.cpu().tolist())
    return perdida_total / total, correctos / total, todas_predicciones, todas_etiquetas


def entrenar(cfg: ConfigEntrenamiento):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizado: {dispositivo}")

    cfg.directorio_modelos.mkdir(parents=True, exist_ok=True)

    cargador_entren, cargador_val, cargador_test = obtener_cargadores(
        cfg.directorio_datos / cfg.archivo_entrenamiento,
        cfg.directorio_datos / cfg.archivo_test,
        batch_size=cfg.batch_size,
        proporcion_validacion=cfg.proporcion_validacion,
    )

    modelo = ClasificadorECG(num_clases=cfg.num_clases, tamano_entrada=cfg.tamano_entrada).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.tasa_aprendizaje, weight_decay=cfg.weight_decay)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=cfg.epocas)

    ejecucion = wandb.init(
        project=cfg.wandb_proyecto,
        entity=cfg.wandb_entidad,
        config={
            "epocas": cfg.epocas,
            "batch_size": cfg.batch_size,
            "tasa_aprendizaje": cfg.tasa_aprendizaje,
            "weight_decay": cfg.weight_decay,
            "arquitectura": "CNN-1D",
            "dataset": "MIT-BIH Arritmia",
        },
    )
    wandb.watch(modelo, log="all", log_freq=100)

    mejor_precision_val = 0.0
    for epoca in range(1, cfg.epocas + 1):
        modelo.train()
        perdida_entren, correctos_entren, total_entren = 0.0, 0, 0

        for X, y in cargador_entren:
            X, y = X.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad()
            logits = modelo(X)
            perdida = criterio(logits, y)
            perdida.backward()
            optimizador.step()

            perdida_entren += perdida.item() * len(y)
            correctos_entren += (logits.argmax(1) == y).sum().item()
            total_entren += len(y)

        planificador.step()

        perdida_val, precision_val, _, _ = evaluar(modelo, cargador_val, criterio, dispositivo)
        precision_entren = correctos_entren / total_entren

        print(
            f"Época {epoca:03d}/{cfg.epocas} | "
            f"perdida_entren={perdida_entren/total_entren:.4f} prec_entren={precision_entren:.4f} | "
            f"perdida_val={perdida_val:.4f} prec_val={precision_val:.4f}"
        )

        wandb.log({
            "epoca": epoca,
            "entrenamiento/perdida": perdida_entren / total_entren,
            "entrenamiento/precision": precision_entren,
            "validacion/perdida": perdida_val,
            "validacion/precision": precision_val,
            "tasa_aprendizaje": planificador.get_last_lr()[0],
        })

        if precision_val > mejor_precision_val:
            mejor_precision_val = precision_val
            torch.save(modelo.state_dict(), cfg.directorio_modelos / cfg.nombre_modelo)
            print(f"  → Mejor modelo guardado (prec_val={precision_val:.4f})")

    # Evaluación final en el conjunto de test
    modelo.load_state_dict(torch.load(cfg.directorio_modelos / cfg.nombre_modelo, map_location=dispositivo))
    perdida_test, precision_test, predicciones, etiquetas = evaluar(modelo, cargador_test, criterio, dispositivo)
    print(f"\nTest  perdida={perdida_test:.4f}  precision={precision_test:.4f}")
    print(classification_report(
        etiquetas, predicciones,
        target_names=["Normal", "Supraventricular", "Ventricular", "Fusion", "Desconocido"]
    ))

    wandb.log({
        "test/perdida": perdida_test,
        "test/precision": precision_test,
        "test/matriz_confusion": wandb.plot.confusion_matrix(
            probs=None, y_true=etiquetas, preds=predicciones,
            class_names=["Normal", "Supraventricular", "Ventricular", "Fusion", "Desconocido"],
        ),
    })

    artefacto = wandb.Artifact("ecg-modelo", type="model")
    artefacto.add_file(str(cfg.directorio_modelos / cfg.nombre_modelo))
    ejecucion.log_artifact(artefacto)
    ejecucion.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar el clasificador de latidos ECG")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = ConfigEntrenamiento()
    if args.epocas:
        cfg.epocas = args.epocas
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.tasa_aprendizaje = args.lr

    entrenar(cfg)
