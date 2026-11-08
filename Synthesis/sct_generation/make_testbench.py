"""Construit le banc d'essai : métriques de chaque sCT contre le vrai CT, et coupes rendues en PNG.

    uv run python make_testbench.py                     # patients de test, rendu « gray »
    uv run python make_testbench.py --split val         # patients de val (mise au point du juge)
    uv run python make_testbench.py --style rgb3        # autre style de rendu (voir config.py)
    uv run python make_testbench.py --max-patients 2    # essai rapide sur deux patients

Le script rassemble tous les sCT déjà produits pour les patients de la partie demandée :

  - ceux du U-Net, pour chaque run et chaque checkpoint (predict.py) ;
  - ceux des modèles pré-entraînés Li, Fu et Spadea (pretrained.py).

Pour chaque patient, il prend les mêmes coupes axiales (SLICE_POSITIONS) dans tous les sCT,
les recadre sur le corps et les enregistre en PNG, sans aucun texte. Il écrit dans
work/testbench/<split>/<style>/ :

    renders/<item_id>.png   les coupes
    testbench.csv           une ligne par PNG

Colonnes de testbench.csv :

    item_id, image_path           les deux seules colonnes que lit benchmark.py
    split, patient_id, center     d'où vient la coupe
    generator                     le modèle qui a produit le sCT : unet_baseline/epoch_005, fedsynth/li...
    slice_index, slice_position   la coupe axiale (indice dans le volume, position relative)
    render_style                  le style de rendu
    mae_hu, psnr_db, ssim         métriques sur le volume entier
    mae_hu_slice, psnr_db_slice, ssim_slice
                                  métriques sur la coupe rendue seule : c'est ce que le VLM voit
    maybe_seen_in_training        1 si le générateur a peut-être été entraîné sur ce patient

Les métriques du volume entier (le calcul le plus long, surtout la SSIM 3D) sont gardées dans
work/testbench/<split>/volume_metrics.csv : changer de style de rendu ne les recalcule pas.

Le VLM ne verra que les PNG. Les métriques restent dans le CSV et ne servent qu'à l'analyse.
"""

import argparse
import csv
import os

import numpy as np
import SimpleITK as sitk

import config
import metrics
import render
from data import read_split

PRETRAINED_PREFIX = "fedsynth"

VOLUME_COLUMNS = ["patient_id", "generator", "sct_mtime", "mae_hu", "psnr_db", "ssim"]
TESTBENCH_COLUMNS = [
    "item_id", "image_path", "split", "patient_id", "center", "generator",
    "slice_index", "slice_position", "render_style",
    "mae_hu", "psnr_db", "ssim", "mae_hu_slice", "psnr_db_slice", "ssim_slice",
    "maybe_seen_in_training",
]


# --- Trouver les sCT ------------------------------------------------------------

def find_scts(split, patient_ids):
    """Renvoie {patient_id: [(generator, chemin du sCT), ...]} pour tous les sCT disponibles."""
    found = {patient_id: [] for patient_id in patient_ids}

    # U-Net : work/runs/<run>/predictions/<split>/epoch_XXX/<patient>/sct.nii.gz
    for run_dir in sorted(config.RUNS_DIR.glob("*")):
        for epoch_dir in sorted((run_dir / "predictions" / split).glob("epoch_*")):
            for patient_id in patient_ids:
                path = epoch_dir / patient_id / "sct.nii.gz"
                if path.exists():
                    found[patient_id].append((f"{run_dir.name}/{epoch_dir.name}", path))

    # Modèles pré-entraînés : work/pretrained/<split>/<patient>/<patient>_<modèle>.synct.nii.gz
    for patient_id in patient_ids:
        for model in config.PRETRAINED_MODELS:
            path = config.PRETRAINED_DIR / split / patient_id / f"{patient_id}_{model}.synct.nii.gz"
            if path.exists():
                found[patient_id].append((f"{PRETRAINED_PREFIX}/{model}", path))

    return found


def center_of(patient_id):
    """'1BA001' -> 'A' (même règle que make_split.py)."""
    return patient_id[2] if len(patient_id) > 2 else "?"


def maybe_seen(generator, patient_id, split):
    """1 si le générateur a peut-être vu ce patient pendant son entraînement."""
    if generator.startswith(PRETRAINED_PREFIX + "/"):
        return int(center_of(patient_id) in config.PRETRAINED_SEEN_CENTERS)
    # Le U-Net n'est entraîné que sur la partie train.
    return int(split == "train")


# --- Lire les volumes -------------------------------------------------------------

def read_image(path):
    """Lit un volume et le met dans l'orientation standard LPS.

    Ainsi, une coupe axiale s'affiche comme en radiologie : avant du crâne en haut,
    côté droit du patient à gauche de l'image.
    """
    return sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")


def same_grid(image, reference):
    return (
        image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing(), atol=1e-3)
        and np.allclose(image.GetOrigin(), reference.GetOrigin(), atol=1e-2)
        and np.allclose(image.GetDirection(), reference.GetDirection(), atol=1e-4)
    )


def read_on_grid(path, reference):
    """Lit un sCT et renvoie son tableau (z, y, x) en HU, sur la grille du vrai CT.

    Normalement tous les sCT sont déjà sur cette grille (celle de l'IRM, qui est aussi celle
    du CT dans SynthRAD). Sinon, on rééchantillonne plutôt que de comparer des voxels qui ne
    se correspondent pas.
    """
    image = read_image(path)
    if not same_grid(image, reference):
        print(f"    {path.name} n'est pas sur la grille du CT : rééchantillonnage.")
        image = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear,
                              float(config.CT_MIN_HU), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(image).astype(np.float32)


def choose_slices(mask):
    """Indices des coupes axiales à rendre, d'après SLICE_POSITIONS et l'étendue du corps."""
    body_slices = np.flatnonzero(mask.any(axis=(1, 2)))
    if body_slices.size == 0:
        return []
    first, last = body_slices[0], body_slices[-1]
    return [int(round(first + position * (last - first))) for position in config.SLICE_POSITIONS]


# --- Cache des métriques du volume entier -----------------------------------------

def read_volume_cache(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(row["patient_id"], row["generator"]): row for row in csv.DictReader(f)}


def append_volume_cache(path, row):
    is_new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VOLUME_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def volume_metrics(cache, cache_path, patient_id, generator, sct_path, sct, ct, mask):
    """Métriques du volume entier, lues dans le cache si le sCT n'a pas changé depuis."""
    mtime = str(round(os.path.getmtime(sct_path)))
    cached = cache.get((patient_id, generator))
    if cached and cached["sct_mtime"] == mtime:
        return {name: float(cached[name]) for name in ("mae_hu", "psnr_db", "ssim")}

    values = metrics.all_metrics(sct, ct, mask)
    row = {"patient_id": patient_id, "generator": generator, "sct_mtime": mtime, **values}
    append_volume_cache(cache_path, row)
    cache[(patient_id, generator)] = row
    return values


# --- Programme principal ------------------------------------------------------------

def process_patient(patient_id, scts, split, style_name, renders_dir, cache, cache_path):
    """Calcule les métriques et rend les coupes de tous les sCT d'un patient. Renvoie les lignes du CSV."""
    style = config.RENDER_STYLES[style_name]
    patient_dir = config.SYNTHRAD_DIR / patient_id
    ct_image = read_image(patient_dir / config.CT_FILE)
    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32)
    mask = read_on_grid(patient_dir / config.MASK_FILE, ct_image) > 0.5

    spacing_x, spacing_y = ct_image.GetSpacing()[:2]
    if not np.isclose(spacing_x, spacing_y, rtol=0.01):
        print(f"    Attention : pixels non carrés ({spacing_x:.2f} × {spacing_y:.2f} mm), l'image sera déformée.")

    slices = choose_slices(mask)
    rows = []
    for generator, sct_path in scts:
        sct = read_on_grid(sct_path, ct_image)
        volume = volume_metrics(cache, cache_path, patient_id, generator, sct_path, sct, ct, mask)

        for position, z in zip(config.SLICE_POSITIONS, slices):
            item_id = f"{patient_id}_{generator.replace('/', '-')}_z{z:03d}_{style_name}"
            image_path = renders_dir / f"{item_id}.png"

            displayed = np.where(mask[z], sct[z], config.CT_MIN_HU) if config.MASK_OUTSIDE_BODY else sct[z]
            box = render.body_box(mask[z], config.CROP_MARGIN)
            cropped = render.crop(displayed, box, fill=config.CT_MIN_HU)
            render.render_slice(cropped, style).save(image_path)

            rows.append({
                "item_id": item_id,
                "image_path": image_path.relative_to(renders_dir.parent).as_posix(),
                "split": split,
                "patient_id": patient_id,
                "center": center_of(patient_id),
                "generator": generator,
                "slice_index": z,
                "slice_position": position,
                "render_style": style_name,
                **volume,
                **metrics.all_metrics(sct[z], ct[z], mask[z], suffix="_slice"),
                "maybe_seen_in_training": maybe_seen(generator, patient_id, split),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Métriques et rendu PNG des sCT : construit testbench.csv.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--style", default=config.DEFAULT_RENDER_STYLE, choices=config.RENDER_STYLES)
    parser.add_argument("--max-patients", type=int, help="limiter le nombre de patients (pour un essai)")
    args = parser.parse_args()

    if not config.SYNTHRAD_DIR.is_dir():
        raise SystemExit(f"Dossier introuvable : {config.SYNTHRAD_DIR} (voir SYNTHRAD_DIR dans config.py)")

    patient_ids = read_split(args.split)
    found = find_scts(args.split, patient_ids)
    patient_ids = [p for p in patient_ids if found[p]]
    if not patient_ids:
        raise SystemExit(
            f"Aucun sCT trouvé pour la partie '{args.split}'.\n"
            f"Lancer d'abord predict.py --split {args.split} et/ou pretrained.py --split {args.split}."
        )
    if args.max_patients:
        patient_ids = patient_ids[:args.max_patients]

    output_dir = config.TESTBENCH_DIR / args.split / args.style
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.TESTBENCH_DIR / args.split / "volume_metrics.csv"
    cache = read_volume_cache(cache_path)

    generators = sorted({generator for p in patient_ids for generator, _ in found[p]})
    print(f"{len(patient_ids)} patients ({args.split}), {len(generators)} générateurs : {', '.join(generators)}")
    print(f"Rendu '{args.style}', {len(config.SLICE_POSITIONS)} coupes par sCT.\n")

    rows = []
    for i, patient_id in enumerate(patient_ids, start=1):
        patient_rows = process_patient(patient_id, found[patient_id], args.split, args.style,
                                       renders_dir, cache, cache_path)
        rows += patient_rows
        print(f"[{i}/{len(patient_ids)}] {patient_id} : {len(found[patient_id])} sCT, {len(patient_rows)} images")

    # On écrit d'abord dans un fichier temporaire : si le script est interrompu pendant
    # l'écriture, l'ancien testbench.csv reste intact.
    testbench_csv = output_dir / "testbench.csv"
    temporary = testbench_csv.with_suffix(".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TESTBENCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(testbench_csv)

    print(f"\n{len(rows)} images. Banc d'essai : {testbench_csv}")
    print("MAE moyenne du volume par générateur (HU) :")
    for generator in generators:
        values = {(r["patient_id"]): r["mae_hu"] for r in rows if r["generator"] == generator}
        if values:
            print(f"  {generator:<28} {np.mean(list(values.values())):7.1f}   ({len(values)} patients)")


if __name__ == "__main__":
    main()
