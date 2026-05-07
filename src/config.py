from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

@dataclass
class TrainConfig:
    # Data
    data_dir: Path = ROOT_DIR / "data"
    train_file: str = "mitbih_train.csv"
    test_file: str = "mitbih_test.csv"

    # Model
    num_classes: int = 5
    input_size: int = 187

    # Training
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1

    # Checkpointing
    model_dir: Path = ROOT_DIR / "models"
    model_name: str = "ecg_classifier.pt"

    # W&B
    wandb_project: str = "ecg-heartbeat-mlops"
    wandb_entity: str | None = None  # set to your W&B username if needed

CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}
