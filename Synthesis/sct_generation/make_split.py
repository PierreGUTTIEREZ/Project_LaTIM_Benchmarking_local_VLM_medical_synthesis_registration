"""Découpe les patients SynthRAD2023 en train / val / test et enregistre le résultat dans un CSV.

Dans SynthRAD2023, chaque patient a son propre dossier, par exemple :

    brain/1BA001/mr.nii.gz
                 ct.nii.gz
                 mask.nii.gz

Le nom du dossier code la tâche (1 = IRM -> CT, 2 = CBCT -> CT), la région (B = brain,
P = pelvis), le centre hospitalier (A, B ou C) et le numéro du patient.

Le découpage se fait par patient, et centre par centre : chaque centre est réparti dans les
mêmes proportions entre train, val et test. Sans ça, un centre pourrait se retrouver
uniquement dans le test, et le modèle serait évalué sur des images d'un scanner qu'il
n'a jamais vu.

Le dossier des données (SYNTHRAD_DIR) et le fichier de sortie (SPLIT_CSV) sont lus dans
config.py :

    uv run python make_split.py

Le CSV produit ressemble à :

    patient_id,center,split
    1BA001,A,train
    1BA005,A,test
    ...

Ce fichier est versionné avec git pour que toute l'équipe utilise le même découpage.
"""

import argparse
import csv
import random
from collections import defaultdict

import config


def find_patients(data_dir):
    """Renvoie les identifiants des patients, c'est-à-dire les sous-dossiers qui contiennent un CT."""
    patients = []
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and (folder / config.CT_FILE).exists():
            patients.append(folder.name)
    return patients


def center_of(patient_id):
    """'1BA001' -> 'A'. Le centre est la troisième lettre de l'identifiant."""
    return patient_id[2] if len(patient_id) > 2 else "?"


def split_patients(patients, val_ratio, test_ratio, seed):
    """Renvoie un dict {patient_id: 'train' | 'val' | 'test'}, en répartissant chaque centre."""
    by_center = defaultdict(list)
    for patient in patients:
        by_center[center_of(patient)].append(patient)

    rng = random.Random(seed)
    assignment = {}
    for center in sorted(by_center):
        group = by_center[center]
        rng.shuffle(group)

        n_test = round(len(group) * test_ratio)
        n_val = round(len(group) * val_ratio)

        for patient in group[:n_test]:
            assignment[patient] = "test"
        for patient in group[n_test:n_test + n_val]:
            assignment[patient] = "val"
        for patient in group[n_test + n_val:]:
            assignment[patient] = "train"

    return assignment


def main():
    parser = argparse.ArgumentParser(description="Découpage train / val / test de SynthRAD2023.")
    parser.add_argument("--val", type=float, default=0.15, help="part des patients en val")
    parser.add_argument("--test", type=float, default=0.15, help="part des patients en test")
    args = parser.parse_args()

    data_dir = config.SYNTHRAD_DIR
    output = config.SPLIT_CSV

    if not data_dir.is_dir():
        raise SystemExit(f"Dossier introuvable : {data_dir} (voir SYNTHRAD_DIR dans config.py)")

    patients = find_patients(data_dir)
    if not patients:
        raise SystemExit(f"Aucun dossier patient contenant {config.CT_FILE} dans {data_dir}")

    # On ne réécrit pas un découpage existant : changer de split en cours de projet
    # rendrait les résultats déjà obtenus incomparables avec les nouveaux.
    if output.exists():
        raise SystemExit(f"{output} existe déjà. Le supprimer à la main pour refaire le découpage.")

    assignment = split_patients(patients, args.val, args.test, config.SEED)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "center", "split"])
        for patient in patients:
            writer.writerow([patient, center_of(patient), assignment[patient]])

    print(f"{len(patients)} patients découpés, enregistré dans {output}")
    for split in ("train", "val", "test"):
        members = [p for p in patients if assignment[p] == split]
        centers = sorted({center_of(p) for p in members})
        print(f"  {split:<5} : {len(members):3d} patients (centres {', '.join(centers)})")


if __name__ == "__main__":
    main()
