"""Pré-traite une fois pour toutes les volumes SynthRAD2023 utilisés par le split.

Pour chaque patient, on lit la source (IRM), le CT et le masque du corps, on les normalise
dans [0, 1] et on les enregistre en .npy dans work/cache/<patient>/.

Pourquoi ne pas lire directement les .nii.gz pendant l'entraînement ? Un fichier .nii.gz est
compressé : le décompresser prend du temps, et il faudrait le faire à chaque coupe tirée.
Un .npy se lit coupe par coupe, presque instantanément.

Le script saute les patients déjà pré-traités : on peut le relancer sans risque.

    python preprocess.py
"""

import numpy as np
import SimpleITK as sitk

import config
from data import normalize_ct, normalize_source, read_split


def read_volume(path):
    """Lit une image médicale et renvoie un tableau numpy de forme (z, y, x)."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def preprocess_patient(patient_id):
    patient_dir = config.SYNTHRAD_DIR / patient_id
    source = read_volume(patient_dir / config.SOURCE_FILE)
    ct = read_volume(patient_dir / config.CT_FILE)
    mask = read_volume(patient_dir / config.MASK_FILE)

    output_dir = config.CACHE_DIR / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # float16 divise la place sur disque par deux. Sur la plage CT, l'arrondi reste
    # inférieur à 1 HU, négligeable devant l'erreur d'un sCT (plusieurs dizaines de HU).
    np.save(output_dir / "source.npy", normalize_source(source, mask).astype(np.float16))
    np.save(output_dir / "ct.npy", normalize_ct(ct).astype(np.float16))
    # Le masque est écrit en dernier : s'il existe, le patient est entièrement pré-traité.
    np.save(output_dir / "mask.npy", (mask > 0).astype(np.uint8))


def main():
    if not config.SYNTHRAD_DIR.is_dir():
        raise SystemExit(f"Dossier introuvable : {config.SYNTHRAD_DIR} (voir SYNTHRAD_DIR dans config.py)")

    patients = read_split("train") + read_split("val") + read_split("test")
    for i, patient_id in enumerate(patients, start=1):
        if (config.CACHE_DIR / patient_id / "mask.npy").exists():
            print(f"[{i}/{len(patients)}] {patient_id} : déjà fait")
            continue
        preprocess_patient(patient_id)
        print(f"[{i}/{len(patients)}] {patient_id}")

    print(f"Pré-traitement terminé : {config.CACHE_DIR}")


if __name__ == "__main__":
    main()
