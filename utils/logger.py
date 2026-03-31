from __future__ import annotations

from pathlib import Path
import csv


class TrainingLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "train_log.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        if self.csv_path.exists():
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "mIoU", "Drivable_IoU", "Pixel_Acc", "lr"])

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: dict[str, float], lr: float) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{metrics['mIoU']:.4f}",
                f"{metrics['Drivable_IoU']:.4f}",
                f"{metrics['Pixel_Acc']:.4f}",
                f"{lr:.8f}",
            ])
