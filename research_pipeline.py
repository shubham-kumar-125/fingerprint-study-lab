"""Research-only fingerprint image classifier.

This program exists to teach a careful ML workflow. It does not determine blood
type, and its output must never be used for medical, donor, or transfusion use.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABELS = ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class Record:
    path: Path
    label: str
    participant_id: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(data_dir: Path) -> list[Record]:
    manifest = data_dir / "metadata.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing {manifest}. Read data/README.md and create a consented, "
            "participant-labelled research manifest before training."
        )
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "label", "participant_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("metadata.csv needs relative_path,label,participant_id columns.")
        records = []
        for row in reader:
            label = (row["label"] or "").strip()
            participant = (row["participant_id"] or "").strip()
            image_path = data_dir / (row["relative_path"] or "")
            if label not in LABEL_TO_INDEX:
                raise ValueError(f"Unsupported label {label!r}; expected one of {LABELS}.")
            if not participant:
                raise ValueError("Every record needs a participant_id for leakage-safe splitting.")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in manifest was not found: {image_path}")
            records.append(Record(image_path, label, participant))
    if len(records) < 16:
        raise ValueError("At least 16 records are required for a meaningful demonstration split.")
    return records


def split_by_participant(records: Sequence[Record], seed: int) -> tuple[list[Record], list[Record], list[Record]]:
    indices = np.arange(len(records))
    groups = [record.participant_id for record in records]
    labels = [record.label for record in records]
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_idx, test_idx = next(first.split(indices, labels, groups))
    train_val = [records[index] for index in train_val_idx]
    test = [records[index] for index in test_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    nested_indices = np.arange(len(train_val))
    nested_groups = [record.participant_id for record in train_val]
    nested_labels = [record.label for record in train_val]
    train_idx, validation_idx = next(second.split(nested_indices, nested_labels, nested_groups))
    return ([train_val[index] for index in train_idx], [train_val[index] for index in validation_idx], test)


class FingerprintDataset(Dataset):
    def __init__(self, records: Sequence[Record], transform: transforms.Compose) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, LABEL_TO_INDEX[record.label]


def make_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    records = load_records(Path(args.data_dir))
    train_records, validation_records, test_records = split_by_participant(records, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.96, 1.04)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_loader = DataLoader(FingerprintDataset(train_records, train_transform), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(FingerprintDataset(validation_records, eval_transform), batch_size=args.batch_size)
    test_loader = DataLoader(FingerprintDataset(test_records, eval_transform), batch_size=args.batch_size)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    best_state, best_f1 = None, -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
        validation = evaluate_loader(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_records), "validation_macro_f1": validation["macro_f1"]})
        print(f"Epoch {epoch:02d}: loss={history[-1]['train_loss']:.4f} validation_macro_f1={validation['macro_f1']:.4f}")
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    model.load_state_dict(best_state)
    report = evaluate_loader(model, test_loader, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "warning": "Research-only. Do not use this output to infer or confirm blood type.",
        "split": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
        "participant_counts": {"train": len({r.participant_id for r in train_records}), "validation": len({r.participant_id for r in validation_records}), "test": len({r.participant_id for r in test_records})},
        "history": history, "test": report,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({
        "state_dict": best_state,
        "labels": LABELS,
        "held_out_test_accuracy": report["accuracy"],
        "held_out_test_macro_f1": report["macro_f1"],
        "warning": "Research-only; not a blood typing result.",
    }, output / "model.pt")
    print(f"Saved research checkpoint and metrics to {output.resolve()}")


@torch.inference_mode()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    truth, predictions, confidences = [], [], []
    for images, targets in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
        confidence, prediction = probabilities.max(dim=1)
        truth.extend(targets.tolist())
        predictions.extend(prediction.tolist())
        confidences.extend(confidence.tolist())
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=range(len(LABELS))).tolist(),
        "classification_report": classification_report(truth, predictions, labels=range(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0),
        "confidence_summary": {"mean": float(np.mean(confidences)), "min": float(np.min(confidences)), "max": float(np.max(confidences))},
    }


def predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, index = probabilities.max(dim=0)
    print("RESEARCH-ONLY OUTPUT — NOT A BLOOD-TYPE RESULT")
    print(f"Experimental dataset class: {LABELS[index.item()]} | score: {confidence.item():.3f}")
    print("Do not use this output for diagnosis, donation, transfusion, or any medical decision.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Research-only fingerprint image classification baseline.")
    subparsers = result.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="Train and evaluate a participant-split research baseline.")
    training.add_argument("--data-dir", default="data")
    training.add_argument("--output-dir", default="outputs/run-001")
    training.add_argument("--epochs", type=int, default=12)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--seed", type=int, default=42)
    training.set_defaults(func=train)
    inference = subparsers.add_parser("predict", help="Inspect an experimental checkpoint locally.")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--image", required=True)
    inference.set_defaults(func=predict)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
