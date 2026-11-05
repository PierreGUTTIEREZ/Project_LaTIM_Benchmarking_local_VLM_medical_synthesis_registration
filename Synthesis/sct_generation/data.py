"""Normalisation des images, lecture des volumes pré-traités et tirage des coupes d'entraînement."""

import csv

import numpy as np
import torch

import config


def read_split(split):
    """Identifiants des patients d'une partie : 'train', 'val' ou 'test'."""
    if not config.SPLIT_CSV.exists():
        raise SystemExit(f"{config.SPLIT_CSV} introuvable : lancer d'abord make_split.py.")

    with open(config.SPLIT_CSV, newline="", encoding="utf-8") as f:
        return [row["patient_id"] for row in csv.DictReader(f) if row["split"] == split]


def normalize_source(source, mask):
    """Ramène l'image source dans [0, 1].

    Les intensités IRM n'ont pas d'unité fixe : elles changent d'un patient et d'une machine
    à l'autre. On divise donc par le 99,5e percentile mesuré dans le corps, ce qui rend les
    patients comparables entre eux sans être gêné par quelques voxels très brillants.
    """
    scale = np.percentile(source[mask > 0], 99.5)
    return np.clip(source / scale, 0, 1)


def normalize_ct(ct):
    """HU -> [0, 1]."""
    ct = np.clip(ct, config.CT_MIN_HU, config.CT_MAX_HU)
    return (ct - config.CT_MIN_HU) / (config.CT_MAX_HU - config.CT_MIN_HU)


def to_hu(ct):
    """[0, 1] -> HU. Inverse de normalize_ct."""
    return ct * (config.CT_MAX_HU - config.CT_MIN_HU) + config.CT_MIN_HU


def load_patient(patient_id):
    """Volumes pré-traités d'un patient : {'source', 'ct', 'mask'}, chacun de forme (z, y, x).

    mmap_mode="r" : les fichiers ne sont pas chargés en entier en mémoire, seules les coupes
    qu'on lit le sont. Indispensable quand on a plus de 100 patients.
    """
    folder = config.CACHE_DIR / patient_id
    return {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in ("source", "ct", "mask")}


def random_crop(source, ct, size):
    """Recadre la même zone aléatoire de size × size dans les deux coupes.

    Si la coupe est plus petite que size, on la complète d'abord avec des zéros
    (c'est-à-dire du noir pour la source et de l'air, -1024 HU, pour le CT).
    """
    pad_h = max(size - source.shape[0], 0)
    pad_w = max(size - source.shape[1], 0)
    padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
    source = np.pad(source, padding)
    ct = np.pad(ct, padding)

    top = torch.randint(source.shape[0] - size + 1, (1,)).item()
    left = torch.randint(source.shape[1] - size + 1, (1,)).item()
    return source[top:top + size, left:left + size], ct[top:top + size, left:left + size]


class SliceDataset(torch.utils.data.Dataset):
    """Paires (coupe source, coupe CT) tirées au hasard parmi les patients donnés.

    Une « époque » vaut ici `slices_per_epoch` coupes tirées au hasard, et non un passage
    sur toutes les coupes : la durée d'une époque ne dépend donc pas du nombre de patients.
    """

    def __init__(self, patient_ids, slices_per_epoch, patch_size):
        self.patients = [load_patient(patient_id) for patient_id in patient_ids]
        self.slices_per_epoch = slices_per_epoch
        self.patch_size = patch_size

        # On ne tire que des coupes qui contiennent le patient, pas les coupes vides aux extrémités.
        self.body_slices = [np.flatnonzero(p["mask"].any(axis=(1, 2))) for p in self.patients]

    def __len__(self):
        return self.slices_per_epoch

    def __getitem__(self, index):
        # `index` ne sert pas : chaque appel tire une nouvelle coupe au hasard.
        # On utilise le générateur de torch plutôt que celui de numpy, car torch donne
        # une graine différente à chaque processus du DataLoader (avec numpy, tous les
        # processus tireraient les mêmes coupes).
        p = torch.randint(len(self.patients), (1,)).item()
        slices = self.body_slices[p]
        z = slices[torch.randint(len(slices), (1,)).item()]

        source = np.asarray(self.patients[p]["source"][z], dtype=np.float32)
        ct = np.asarray(self.patients[p]["ct"][z], dtype=np.float32)
        source, ct = random_crop(source, ct, self.patch_size)

        # Le modèle attend des images de forme (canal, hauteur, largeur), avec un seul canal.
        return torch.from_numpy(source)[None], torch.from_numpy(ct)[None]
