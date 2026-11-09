"""Génère les sCT des trois modèles pré-entraînés FedSynthCT-Brain : Li, Fu et Spadea.

Ces modèles viennent de SlicerModalityConverter (Raggio et al., 2025). Vincent les a installés
sur la station, avec un script qui les lance sur les données SynthRAD (guide :
/home/student/SYNTHETIC_CT/GUIDE.md). Ils tournent dans son environnement Python à lui :
ce script ne fait qu'appeler le sien, patient par patient, pour une partie du split.

    uv run python pretrained.py                              # patients de test
    uv run python pretrained.py --split val                  # patients de val
    uv run python pretrained.py --patients 1BA001 1BA005     # quelques patients de test
    uv run python pretrained.py --split val --max-patients 2 # les 2 premiers patients de val (essai)

Pour chaque patient, le script de Vincent écrit dans work/pretrained/<split>/<patient>/ :

    <patient>_li.synct.nii.gz        les trois sCT, en HU, sur la même grille que l'IRM
    <patient>_fu.synct.nii.gz
    <patient>_spadea.synct.nii.gz
    results_summary.csv              sa MAE contre le vrai CT (make_testbench.py recalcule
                                     les métriques de son côté, pareil pour tous les sCT)
    figures/<patient>_montage.png    IRM / Li / Fu / Spadea / CT côte à côte, pour vérifier à l'œil

Les patients déjà faits (les trois sCT présents) sont sautés : on peut relancer sans risque.

Attention : ces modèles ont été entraînés en partie sur le centre C de SynthRAD2023
(voir PRETRAINED_SEEN_CENTERS dans config.py).
"""

import argparse
import subprocess

import config
from data import read_split


def expected_outputs(patient_id, output_dir):
    """Chemins des trois sCT que le script de Vincent doit produire pour ce patient."""
    return [output_dir / f"{patient_id}_{model}.synct.nii.gz" for model in config.PRETRAINED_MODELS]


def run_patient(patient_id, output_dir, device):
    """Lance le script de Vincent pour un seul patient. Lève une erreur s'il échoue."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.PRETRAINED_PYTHON),
        str(config.PRETRAINED_SCRIPT),
        "--subjects", patient_id,
        "--outdir", str(output_dir),
        "--device", device,
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="sCT des modèles pré-entraînés Li, Fu et Spadea.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--patients", nargs="+", help="par défaut : tous les patients de la partie")
    parser.add_argument("--max-patients", type=int, help="limiter le nombre de patients (pour un essai)")
    parser.add_argument("--device", default=config.PRETRAINED_DEVICE, help="ex. cuda:0 ou cpu")
    args = parser.parse_args()

    for path, name in [(config.PRETRAINED_PYTHON, "PRETRAINED_PYTHON"), (config.PRETRAINED_SCRIPT, "PRETRAINED_SCRIPT")]:
        if not path.exists():
            raise SystemExit(
                f"{path} introuvable (voir {name} dans config.py).\n"
                "Ce script doit être lancé sur la station, où Vincent a installé les modèles."
            )

    split_patients = read_split(args.split)
    patients = args.patients or split_patients
    outside = [p for p in patients if p not in split_patients]
    if outside:
        raise SystemExit(f"Pas dans la partie '{args.split}' : {', '.join(outside)} (voir --split).")
    if args.max_patients:
        # Les mêmes premiers patients que make_testbench.py --max-patients.
        patients = patients[:args.max_patients]

    failures = []
    for i, patient_id in enumerate(patients, start=1):
        output_dir = config.PRETRAINED_DIR / args.split / patient_id
        if all(path.exists() for path in expected_outputs(patient_id, output_dir)):
            print(f"[{i}/{len(patients)}] {patient_id} : déjà fait")
            continue

        print(f"[{i}/{len(patients)}] {patient_id}", flush=True)
        try:
            run_patient(patient_id, output_dir, args.device)
        except subprocess.CalledProcessError as error:
            # Un patient qui échoue ne doit pas arrêter les autres.
            print(f"  ERREUR : le script s'est arrêté avec le code {error.returncode}")
            failures.append(patient_id)
            continue

        missing = [path.name for path in expected_outputs(patient_id, output_dir) if not path.exists()]
        if missing:
            print(f"  ERREUR : fichiers attendus absents : {', '.join(missing)}")
            failures.append(patient_id)

    print(f"sCT des modèles pré-entraînés dans {config.PRETRAINED_DIR / args.split}")
    if failures:
        raise SystemExit(f"{len(failures)} patient(s) en échec : {' '.join(failures)} (relancer pour réessayer).")


if __name__ == "__main__":
    main()
