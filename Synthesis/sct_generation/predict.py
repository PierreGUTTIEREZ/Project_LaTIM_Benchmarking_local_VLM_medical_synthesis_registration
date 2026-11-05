"""Génère les sCT d'une partie du split avec chaque checkpoint d'un run.

    python predict.py --run unet_baseline                  # tous les checkpoints, patients de test
    python predict.py --run unet_baseline --split val      # patients de val (mise au point du juge)
    python predict.py --run unet_baseline --epochs 5 100   # seulement certains checkpoints

Les sCT sont écrits en HU, au format NIfTI, avec la même géométrie (origine, espacement,
orientation) que l'IRM d'origine :

    work/runs/unet_baseline/predictions/test/epoch_005/1BA001/sct.nii.gz

Les sCT déjà générés sont sautés : on peut relancer le script sans risque.
"""

import argparse

import numpy as np
import SimpleITK as sitk
import torch

import config
from data import load_patient, read_split, to_hu
from model import load_checkpoint, predict_volume


def save_sct(sct_hu, patient_id, output_path):
    """Enregistre le sCT en reprenant la géométrie de l'image source du patient."""
    reference = sitk.ReadImage(str(config.SYNTHRAD_DIR / patient_id / config.SOURCE_FILE))
    image = sitk.GetImageFromArray(np.round(sct_hu).astype(np.int16))
    image.CopyInformation(reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Générer des sCT avec les checkpoints d'un run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--epochs", nargs="+", type=int, help="par défaut : tous les checkpoints")
    args = parser.parse_args()

    if not config.SYNTHRAD_DIR.is_dir():
        raise SystemExit(f"Dossier introuvable : {config.SYNTHRAD_DIR} (voir SYNTHRAD_DIR dans config.py)")

    run_dir = config.RUNS_DIR / args.run
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    if args.epochs:
        checkpoints = [c for c in checkpoints if int(c.stem.split("_")[1]) in args.epochs]
    if not checkpoints:
        raise SystemExit(f"Aucun checkpoint trouvé dans {run_dir / 'checkpoints'}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patient_ids = read_split(args.split)

    for checkpoint in checkpoints:
        model = load_checkpoint(checkpoint, device)
        print(f"{checkpoint.stem} : {len(patient_ids)} patients ({args.split})")

        for patient_id in patient_ids:
            output_path = run_dir / "predictions" / args.split / checkpoint.stem / patient_id / "sct.nii.gz"
            if output_path.exists():
                continue
            sct = predict_volume(model, load_patient(patient_id)["source"], device)
            save_sct(to_hu(sct), patient_id, output_path)

    print(f"sCT enregistrés dans {run_dir / 'predictions' / args.split}")


if __name__ == "__main__":
    main()
