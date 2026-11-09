"""Métriques de dose : un plan de traitement calculé sur le sCT donne-t-il la même dose que sur le vrai CT ?

    uv run python dose.py --split val                    # tous les sCT de val, photons et protons
    uv run python dose.py --split val --max-patients 2   # essai rapide
    uv run python dose.py --modality photons             # test, photons seulement

Principe, le même que celui du challenge SynthRAD2023 :

  1. pour chaque patient, on définit une cible sphérique (SynthRAD ne fournit pas de contours,
     voir DOSE_TARGET_RADIUS_MM dans config.py) et on optimise un plan sur le VRAI CT,
     normalisé pour que la dose moyenne dans la cible soit la dose prescrite ;
  2. on recalcule exactement le même plan (mêmes faisceaux, mêmes poids) sur chaque sCT ;
  3. on compare la dose obtenue sur le sCT à celle du vrai CT.

Le calcul de dose est fait par pyRadPlan (portage Python de matRad, DKFZ) : photons 6 MV par
pencil beam SVD, protons par pencil beam. Le gamma est calculé par pymedphys.

Métriques, pour chaque modalité (photons, protons) :

    gamma        taux de réussite gamma 2 %/2 mm, en %                 (plus haut = meilleur)
    dose_diff    écart de dose moyen |sCT - CT| où le CT reçoit plus de 10 % du maximum,
                 en % de la dose prescrite                              (plus bas = meilleur)
    ptv_d98      |écart de D98| dans la cible, en % de la prescription  (plus bas = meilleur)
    ptv_d2       |écart de D2| dans la cible, en % de la prescription   (plus bas = meilleur)
    body_dmean   |écart de dose moyenne| dans le corps, en %            (plus bas = meilleur)

D98 est la dose reçue par au moins 98 % de la cible (sa couverture), D2 la dose reçue par les
2 % les plus irradiés (son point chaud).

Écrit work/testbench/<split>/dose_metrics.csv, une ligne par (patient, générateur, modalité).
analyze.py y prend les métriques automatiquement : il suffit de le relancer, sans refaire le
benchmark. Les sCT déjà traités sont sautés : on peut relancer après une coupure.

Le calcul tourne sur le processeur (DOSE_USE_GPU) : il peut tourner pendant le benchmark des VLM.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # coupe les barres de progression de pyRadPlan

import argparse
import csv
import logging
import time
import warnings

import numpy as np
import SimpleITK as sitk

import config
from data import read_split
from make_testbench import find_scts, read_image, read_on_grid

METRIC_NAMES = ["gamma", "dose_diff", "ptv_d98", "ptv_d2", "body_dmean"]
CACHE_COLUMNS = ["patient_id", "generator", "modality", "sct_mtime", *METRIC_NAMES]


# --- pyRadPlan ------------------------------------------------------------------------

def import_pyradplan():
    """Importe pyRadPlan et le règle (processeur, pas de messages). Importé ici seulement pour
    que les autres scripts du dossier n'en dépendent pas."""
    try:
        import pyRadPlan
    except ImportError:
        raise SystemExit("pyRadPlan n'est pas installé : lancer « uv sync » dans Synthesis/.")
    pyRadPlan.settings.xp.prefer_gpu = config.DOSE_USE_GPU
    pyRadPlan.settings.xp.preferred_cpu_array_backend = "numpy"
    # pyRadPlan est bavard (avertissements de calcul sans conséquence, réglages repris du plan) :
    # on ne garde que les vraies erreurs, pour que la sortie reste lisible.
    logging.getLogger("pyRadPlan").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Resampling created an empty structure")
    return pyRadPlan


def as_image(array, reference):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def make_plan(modality):
    """Plan vide (faisceaux et grille de dose), sans optimisation."""
    from pyRadPlan import IonPlan, PhotonPlan

    angles = np.array(config.DOSE_GANTRY_ANGLES[modality], dtype=float)
    if modality == "photons":
        plan = PhotonPlan(machine="Generic")       # faisceau générique de 6 MV
        engine = "SVDPB"                           # pencil beam par décomposition en valeurs singulières
    else:
        plan = IonPlan(radiation_mode="protons", machine="Generic")
        engine = "HongPB"                          # pencil beam de Hong
    plan.prop_stf = {"gantry_angles": angles, "couch_angles": np.zeros_like(angles)}
    step = config.DOSE_GRID_MM
    plan.prop_dose_calc = {"engine": engine, "dose_grid": {"resolution": {"x": step, "y": step, "z": step}}}
    plan.prop_opt = {"solver": "scipy"}
    return plan


def structures(ct, reference, body, target):
    """Contours au format pyRadPlan : la cible, et le corps entier comme organe à protéger."""
    from pyRadPlan.cst import create_cst, create_voi
    from pyRadPlan.optimization.objectives import MeanDose, SquaredDeviation, SquaredOverdosing

    prescription = config.DOSE_PRESCRIPTION_GY
    ptv = create_voi(voi_type="TARGET", name="PTV", ct_image=ct,
                     mask=as_image(target.astype(np.uint8), reference))
    # Le corps est déclaré « OAR » (et non « EXTERNAL ») : sinon pyRadPlan n'y calcule pas la dose.
    body_voi = create_voi(voi_type="OAR", name="BODY", ct_image=ct,
                          mask=as_image(body.astype(np.uint8), reference))
    ptv.objectives = [SquaredDeviation(priority=100.0, d_ref=prescription)]
    body_voi.objectives = [SquaredOverdosing(priority=10.0, d_max=0.5 * prescription),
                           MeanDose(priority=1.0, d_ref=0.0)]
    return create_cst([ptv, body_voi], ct=ct)


def dose_array(result):
    dose = result["physical_dose"]
    array = sitk.GetArrayFromImage(dose) if isinstance(dose, sitk.Image) else np.asarray(dose)
    return array.astype(np.float32)


def plan_on_ct(ct_hu, reference, body, target, modality):
    """Optimise le plan sur le vrai CT. Renvoie ce qu'il faut pour le recalculer, et la dose de référence."""
    from pyRadPlan import calc_dose_influence, fluence_optimization, generate_stf
    from pyRadPlan.ct import create_ct

    ct = create_ct(cube_hu=as_image(ct_hu, reference))
    cst = structures(ct, reference, body, target)
    plan = make_plan(modality)
    stf = generate_stf(ct, cst, plan)
    dij = calc_dose_influence(ct, cst, stf, plan)
    weights = np.asarray(fluence_optimization(ct, cst, stf, dij, plan))
    dose = dose_array(dij.compute_result_ct_grid(weights))

    # Normalisation : dose moyenne dans la cible = dose prescrite.
    scale = config.DOSE_PRESCRIPTION_GY / float(dose[target].mean())
    return {"cst": cst, "plan": plan, "stf": stf, "weights": weights * scale}, dose * scale


def recompute(plan_data, sct_hu, reference):
    """Recalcule le même plan (mêmes faisceaux, mêmes poids) sur un autre CT."""
    from pyRadPlan import calc_dose_forward
    from pyRadPlan.ct import create_ct
    from pyRadPlan.cst import create_cst

    ct = create_ct(cube_hu=as_image(sct_hu, reference))
    cst = create_cst(plan_data["cst"], ct=ct)
    result = calc_dose_forward(ct, cst, plan_data["stf"], plan_data["plan"], weights=plan_data["weights"])
    return dose_array(result)


# --- Cible et métriques -----------------------------------------------------------------

def spherical_target(ct_hu, body, spacing_zyx):
    """Sphère de DOSE_TARGET_RADIUS_MM centrée sur les tissus mous (0 à 80 HU) du vrai CT."""
    soft = body & (ct_hu > 0) & (ct_hu < 80)
    center = np.argwhere(soft if soft.any() else body).mean(axis=0)
    grid = np.indices(body.shape, dtype=np.float32)
    distance2 = sum(((grid[axis] - center[axis]) * spacing_zyx[axis]) ** 2 for axis in range(3))
    return (distance2 <= config.DOSE_TARGET_RADIUS_MM**2) & body


def gamma_pass_rate(reference_dose, evaluated_dose, reference):
    """Taux de réussite gamma global, en %, sur une grille de GAMMA_GRID_MM."""
    import pymedphys

    step = config.GAMMA_GRID_MM
    size = [int(round(n * s / step)) for n, s in zip(reference.GetSize(), reference.GetSpacing())]
    resampled = []
    for dose in (reference_dose, evaluated_dose):
        image = sitk.Resample(as_image(dose, reference), size, sitk.Transform(), sitk.sitkLinear,
                              reference.GetOrigin(), (step, step, step), reference.GetDirection(), 0.0)
        resampled.append(sitk.GetArrayFromImage(image))
    axes = tuple(np.arange(n) * step for n in resampled[0].shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamma = pymedphys.gamma(
            axes, resampled[0], axes, resampled[1],
            dose_percent_threshold=config.GAMMA_DOSE_PERCENT,
            distance_mm_threshold=config.GAMMA_DISTANCE_MM,
            lower_percent_dose_cutoff=config.GAMMA_CUTOFF_PERCENT,
            max_gamma=2.0, quiet=True,
        )
    valid = np.isfinite(gamma)
    return float(100 * np.mean(gamma[valid] <= 1)) if valid.any() else float("nan")


def compare_doses(reference_dose, evaluated_dose, target, body, reference):
    """Les cinq métriques de dose (voir l'en-tête du fichier)."""
    percent = 100 / config.DOSE_PRESCRIPTION_GY
    irradiated = body & (reference_dose > config.GAMMA_CUTOFF_PERCENT / 100 * reference_dose.max())

    def d(dose, coverage):  # dose reçue par au moins `coverage` % de la cible
        return float(np.percentile(dose[target], 100 - coverage))

    return {
        "gamma": round(gamma_pass_rate(reference_dose, evaluated_dose, reference), 3),
        "dose_diff": round(float(np.abs(evaluated_dose - reference_dose)[irradiated].mean()) * percent, 4),
        "ptv_d98": round(abs(d(evaluated_dose, 98) - d(reference_dose, 98)) * percent, 4),
        "ptv_d2": round(abs(d(evaluated_dose, 2) - d(reference_dose, 2)) * percent, 4),
        "body_dmean": round(abs(float(evaluated_dose[body].mean() - reference_dose[body].mean())) * percent, 4),
    }


# --- Cache ------------------------------------------------------------------------------

def read_cache(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["patient_id"], r["generator"], r["modality"]): r for r in csv.DictReader(f)}


def append_cache(path, row):
    is_new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


# --- Programme principal ------------------------------------------------------------------

def process_patient(patient_id, scts, modalities, split, cache, cache_path):
    """Plan sur le vrai CT puis recalcul sur chaque sCT pas encore traité. Renvoie le nombre de calculs."""
    todo = {
        modality: [(generator, path) for generator, path in scts
                   if cache.get((patient_id, generator, modality), {}).get("sct_mtime")
                   != str(round(os.path.getmtime(path)))]
        for modality in modalities
    }
    if not any(todo.values()):
        return 0

    patient_dir = config.SYNTHRAD_DIR / patient_id
    reference = read_image(patient_dir / config.CT_FILE)
    body = read_on_grid(patient_dir / config.MASK_FILE, reference) > 0.5
    # Hors du corps : de l'air, pour le CT comme pour les sCT (la table et le bruit de fond ne
    # doivent pas changer la dose).
    ct_hu = np.where(body, sitk.GetArrayFromImage(reference), config.CT_MIN_HU).astype(np.float32)
    spacing_zyx = reference.GetSpacing()[::-1]
    target = spherical_target(ct_hu, body, spacing_zyx)

    done = 0
    for modality, pending in todo.items():
        if not pending:
            continue
        start = time.perf_counter()
        plan_data, reference_dose = plan_on_ct(ct_hu, reference, body, target, modality)
        output_dir = config.DOSE_DIR / split / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(as_image(reference_dose, reference), str(output_dir / f"dose_ct_{modality}.nii.gz"))
        coverage = np.percentile(reference_dose[target], 2) / config.DOSE_PRESCRIPTION_GY * 100
        print(f"    {modality} : plan optimisé sur le CT en {time.perf_counter() - start:.0f} s "
              f"(D98 de la cible = {coverage:.0f} % de la prescription)", flush=True)

        for generator, path in pending:
            sct_hu = np.where(body, read_on_grid(path, reference), config.CT_MIN_HU).astype(np.float32)
            values = compare_doses(reference_dose, recompute(plan_data, sct_hu, reference), target, body, reference)
            row = {"patient_id": patient_id, "generator": generator, "modality": modality,
                   "sct_mtime": str(round(os.path.getmtime(path))), **values}
            append_cache(cache_path, row)
            cache[(patient_id, generator, modality)] = row
            print(f"      {generator:<28} gamma {values['gamma']:5.1f} %   "
                  f"écart de dose {values['dose_diff']:.2f} %", flush=True)
            done += 1
    return done


def main():
    parser = argparse.ArgumentParser(description="Métriques de dose (gamma, DVH) des sCT contre le vrai CT.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--modality", choices=config.DOSE_MODALITIES, help="par défaut : toutes")
    parser.add_argument("--max-patients", type=int, help="limiter le nombre de patients (pour un essai)")
    args = parser.parse_args()

    import_pyradplan()
    modalities = [args.modality] if args.modality else config.DOSE_MODALITIES

    patient_ids = read_split(args.split)
    found = find_scts(args.split, patient_ids)
    patient_ids = [p for p in patient_ids if found[p]]
    if not patient_ids:
        raise SystemExit(f"Aucun sCT trouvé pour la partie '{args.split}' : lancer d'abord predict.py et/ou pretrained.py.")
    if args.max_patients:
        patient_ids = patient_ids[:args.max_patients]

    cache_path = config.TESTBENCH_DIR / args.split / "dose_metrics.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = read_cache(cache_path)
    print(f"{len(patient_ids)} patients ({args.split}), modalités : {', '.join(modalities)}. "
          f"Résultats : {cache_path}\n")

    failures = []
    for i, patient_id in enumerate(patient_ids, start=1):
        print(f"[{i}/{len(patient_ids)}] {patient_id} : {len(found[patient_id])} sCT", flush=True)
        try:
            if process_patient(patient_id, found[patient_id], modalities, args.split, cache, cache_path) == 0:
                print("    déjà fait")
        except Exception as error:
            # Un patient qui échoue ne doit pas arrêter les autres. Ctrl+C arrête tout, lui.
            print(f"    ERREUR {type(error).__name__}: {error}")
            failures.append(patient_id)

    print(f"\nMétriques de dose dans {cache_path}. Relancer analyze.py pour les prendre en compte.")
    if failures:
        raise SystemExit(f"{len(failures)} patient(s) en échec : {' '.join(failures)} (relancer pour réessayer).")


if __name__ == "__main__":
    main()
