import subprocess
import sys

EXPERIMENTOS = [
    ###### Experimentos iniciales (20 épocas)
    {"nombre": "cnn-orig-20ep", "arquitectura": "CNN", "dataset": "mitbih:v0", "epocas": 20},
    {"nombre": "cnn-bal-20ep", "arquitectura": "CNN", "dataset": "mitbih:v1", "epocas": 20},
    {"nombre": "gru-orig-20ep", "arquitectura": "GRU", "dataset": "mitbih:v0", "epocas": 20},
    {"nombre": "gru-bal-20ep", "arquitectura": "GRU", "dataset": "mitbih:v1", "epocas": 20},
    ###### (30 épocas)
    {"nombre": "cnn-bal-30ep", "arquitectura": "CNN", "dataset": "mitbih:v1", "epocas": 30},
    {"nombre": "gru-bal-30ep", "arquitectura": "GRU", "dataset": "mitbih:v1", "epocas": 30},
    ###### (lr=5e-4)
    {"nombre": "cnn-bal-25ep-lr5e4", "arquitectura": "CNN", "dataset": "mitbih:v1", "epocas": 25, "lr": 5e-4},
    {"nombre": "gru-bal-25ep-lr5e4", "arquitectura": "GRU", "dataset": "mitbih:v1", "epocas": 25, "lr": 5e-4},
    ###### (15 épocas y weight decay 1e-3)
    {"nombre": "cnn-bal-15ep-wd1e3", "arquitectura": "CNN", "dataset": "mitbih:v1", "epocas": 15, "weight_decay": 1e-3},
    {"nombre": "gru-bal-15ep-wd1e3", "arquitectura": "GRU", "dataset": "mitbih:v1", "epocas": 15, "weight_decay": 1e-3},
]


def construir_comando(exp: dict) -> list[str]:
    cmd = [sys.executable, "src/train.py"]
    for clave, valor in exp.items():
        cmd += [f"--{clave}", str(valor)]
    return cmd


def main():
    total = len(EXPERIMENTOS)
    for i, exp in enumerate(EXPERIMENTOS, start=1):
        print(f"\n{'='*60}")
        print(f"  Experimento {i}/{total}: {exp['nombre']}")
        print(f"{'='*60}\n")

        cmd = construir_comando(exp)
        resultado = subprocess.run(cmd, cwd=".")

        if resultado.returncode != 0:
            print(f"\n[!] El experimento '{exp['nombre']}' falló (código {resultado.returncode}).")
            continuar = input("¿Continuar con el siguiente? [s/N]: ").strip().lower()
            if continuar != "s":
                print("Abortando.")
                sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Todos los experimentos completados.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
