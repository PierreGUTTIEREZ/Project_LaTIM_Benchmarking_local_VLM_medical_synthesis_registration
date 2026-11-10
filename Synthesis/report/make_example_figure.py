"""Figure d'exemples pour le rapport : une même coupe, du vrai CT aux différents sCT, avec les notes des VLM.

À lancer SUR LA STATION (c'est là que se trouvent les PNG du banc d'essai et les données) :

    cd Synthesis/report
    uv run python make_example_figure.py                     # test, rendu gray, 2 patients
    uv run python make_example_figure.py --patients 1BA001 1BC027

Écrit figures/exemples.pdf et figures/exemples.png, que rapport.tex inclut automatiquement s'ils
existent. Rapatrier ensuite ces deux fichiers sur le PC avec scp avant de recompiler le rapport.

Les images proviennent de SynthRAD2023 (licence de recherche, citation obligatoire) : vérifier avec
l'encadrant que leur diffusion dans le rapport est acceptable avant de l'envoyer hors de l'équipe.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sct_generation"))
import config  # noqa: E402  (config de sct_generation)
import render  # noqa: E402
from make_testbench import read_image, read_on_grid  # noqa: E402

COLUMNS = ["unet_baseline/epoch_001", "unet_baseline/epoch_005", "unet_baseline/epoch_020",
           "unet_baseline/epoch_100", "fedsynth/li"]
LABELS = {"unet_baseline/epoch_001": "U-Net ép. 1", "unet_baseline/epoch_005": "U-Net ép. 5",
          "unet_baseline/epoch_020": "U-Net ép. 20", "unet_baseline/epoch_100": "U-Net ép. 100",
          "fedsynth/li": "Li (pré-entraîné)"}
JUDGES = [("qwen3-vl:30b", "Qwen3-VL 30B"), ("medgemma:latest", "MedGemma 4B")]


def real_ct_png(patient_id, z, style):
    """Rend la coupe du vrai CT exactement comme les sCT (même masque, même recadrage)."""
    patient_dir = config.SYNTHRAD_DIR / patient_id
    ct_image = read_image(patient_dir / config.CT_FILE)
    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32)
    mask = read_on_grid(patient_dir / config.MASK_FILE, ct_image) > 0.5
    shown = np.where(mask[z], ct[z], config.CT_MIN_HU)
    box = render.body_box(mask[z], config.CROP_MARGIN)
    return render.render_slice(render.crop(shown, box, fill=config.CT_MIN_HU), config.RENDER_STYLES[style])


def main():
    parser = argparse.ArgumentParser(description="Figure d'exemples (vrai CT et sCT) pour le rapport.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--style", default="gray")
    parser.add_argument("--patients", nargs="+", help="par défaut : les 2 premiers patients du banc d'essai")
    parser.add_argument("--position", type=float, default=0.5, help="position de la coupe (voir SLICE_POSITIONS)")
    args = parser.parse_args()

    testbench_dir = config.TESTBENCH_DIR / args.split / args.style
    testbench = pd.read_csv(testbench_dir / "testbench.csv")
    scores_csv = HERE.parent / "results" / "analysis" / f"{args.split}_{args.style}" / "scores_and_metrics.csv"
    scores = pd.read_csv(scores_csv) if scores_csv.exists() else pd.DataFrame(columns=["item_id", "model", "score"])
    patients = args.patients or sorted(testbench.patient_id.unique())[:2]

    figure, axes = plt.subplots(len(patients), len(COLUMNS) + 1, figsize=(7.0, 1.55 * len(patients) + 0.25),
                                squeeze=False)
    for row, patient_id in zip(axes, patients):
        items = testbench[(testbench.patient_id == patient_id) & (testbench.slice_position == args.position)]
        if items.empty:
            raise SystemExit(f"Pas de coupe à la position {args.position} pour {patient_id}.")
        z = int(items.slice_index.iloc[0])
        row[0].imshow(real_ct_png(patient_id, z, args.style), cmap="gray")
        row[0].set_title(f"CT réel · {patient_id}", fontsize=6.5)
        for ax, generator in zip(row[1:], COLUMNS):
            item = items[items.generator == generator]
            if item.empty:
                ax.set_visible(False)
                continue
            item = item.iloc[0]
            ax.imshow(Image.open(testbench_dir / item.image_path), cmap="gray")
            notes = []
            for model, short in JUDGES:
                s = scores[(scores.item_id == item.item_id) & (scores.model == model)]
                if not s.empty:
                    notes.append(f"{short} : {s.score.iloc[0]:.0f}")
            ax.set_title(f"{LABELS[generator]}\nMAE {item.mae_hu_slice:.0f} HU", fontsize=6.3)
            ax.text(0.5, -0.04, "\n".join(notes), transform=ax.transAxes, ha="center", va="top", fontsize=5.8)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    figure.tight_layout(h_pad=1.6)
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "exemples.pdf", bbox_inches="tight")
    figure.savefig(output / "exemples.png", dpi=200, bbox_inches="tight")
    print(f"Écrit {output / 'exemples.pdf'} : à rapatrier sur le PC (scp) avant de recompiler.")


if __name__ == "__main__":
    main()
