"""Entraîne le U-Net IRM -> CT et sauvegarde des checkpoints à plusieurs époques.

    python train.py --run unet_baseline

Crée work/runs/unet_baseline/ avec :
    checkpoints/epoch_001.pt, epoch_002.pt, ...   (époques de config.CHECKPOINT_EPOCHS)
    training_log.csv                              (perte d'entraînement et MAE de validation)

La MAE de validation (en HU, dans le corps) est calculée à chaque checkpoint sur les patients
entiers de val. C'est elle qui sert à comparer des réglages d'entraînement entre eux ; le test
n'est jamais utilisé ici.
"""

import argparse
import csv
import time

import numpy as np
import torch

import config
from data import SliceDataset, load_patient, read_split, to_hu
from model import build_model, predict_volume


def validation_mae(model, patient_ids, device):
    """MAE moyenne en HU, calculée dans le masque du corps (même métrique que SynthRAD2023)."""
    errors = []
    for patient_id in patient_ids:
        patient = load_patient(patient_id)
        sct = predict_volume(model, patient["source"], device)
        ct = np.asarray(patient["ct"], dtype=np.float32)
        body = patient["mask"] > 0
        errors.append(np.abs(to_hu(sct) - to_hu(ct))[body].mean())
    return float(np.mean(errors))


def train(run_name):
    run_dir = config.RUNS_DIR / run_name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} existe déjà : choisir un autre nom de run.")
    (run_dir / "checkpoints").mkdir(parents=True)

    torch.manual_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entraînement sur {device}.")

    train_ids = read_split("train")
    val_ids = read_split("val")
    print(f"{len(train_ids)} patients d'entraînement, {len(val_ids)} de validation.")

    loader = torch.utils.data.DataLoader(
        SliceDataset(train_ids, config.SLICES_PER_EPOCH, config.PATCH_SIZE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = torch.nn.L1Loss()

    log_path = run_dir / "training_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_l1", "val_mae_hu", "duration_s"])

    for epoch in range(1, config.EPOCHS + 1):
        start = time.perf_counter()
        model.train()
        losses = []
        for source, ct in loader:
            source, ct = source.to(device), ct.to(device)
            loss = loss_function(model(source), ct)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_l1 = float(np.mean(losses))

        val_mae = ""
        if epoch in config.CHECKPOINT_EPOCHS or epoch == config.EPOCHS:
            torch.save(model.state_dict(), run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
            val_mae = round(validation_mae(model, val_ids, device), 1)

        duration = round(time.perf_counter() - start, 1)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_l1, 5), val_mae, duration])

        message = f"Époque {epoch:3d}/{config.EPOCHS}  L1 {train_l1:.4f}  ({duration} s)"
        if val_mae != "":
            message += f"  checkpoint sauvegardé, MAE val {val_mae} HU"
        print(message)

    print(f"Entraînement terminé : {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Entraîner le U-Net IRM -> CT.")
    parser.add_argument("--run", required=True, help="nom du run, ex. unet_baseline")
    args = parser.parse_args()
    train(args.run)


if __name__ == "__main__":
    main()
