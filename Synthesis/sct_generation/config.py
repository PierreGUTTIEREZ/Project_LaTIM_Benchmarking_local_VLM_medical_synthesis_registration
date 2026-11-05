"""Paramètres de la génération des sCT (IRM -> CT)."""

from pathlib import Path

SCT_DIR = Path(__file__).resolve().parent

# --- Données ----------------------------------------------------------------

# Dossier d'une région de SynthRAD2023 (Task1, IRM -> CT) sur latim5090.
# Il contient directement les dossiers patients : 1BA001/, 1BA005/, ...
SYNTHRAD_DIR = Path("/home/student/DATA/SYNTHESIS/brain")

# Découpage produit par make_split.py pour cette même région.
SPLIT_CSV = SCT_DIR / "splits" / "task1_brain.csv"

# Noms des fichiers dans chaque dossier patient.
# On appelle « source » l'image d'entrée : l'IRM pour la Task1, le CBCT pour la Task2.
SOURCE_FILE = "mr.nii.gz"  # "cbct.nii.gz" pour la Task2
CT_FILE = "ct.nii.gz"
MASK_FILE = "mask.nii.gz"

# Tout ce qui est dérivé des images patient (volumes pré-traités, checkpoints, sCT)
# va dans work/, qui n'est pas versionné.
WORK_DIR = SCT_DIR / "work"
CACHE_DIR = WORK_DIR / "cache"
RUNS_DIR = WORK_DIR / "runs"

# Plage de HU conservée pour le CT (même plage que le challenge SynthRAD2023).
CT_MIN_HU = -1024
CT_MAX_HU = 3000

# --- Entraînement -----------------------------------------------------------

PATCH_SIZE = 256           # taille des coupes recadrées vues pendant l'entraînement
BATCH_SIZE = 16
SLICES_PER_EPOCH = 2000    # une époque = 2000 coupes tirées au hasard
EPOCHS = 100
LEARNING_RATE = 2e-4
NUM_WORKERS = 4            # processus qui préparent les batchs en parallèle (0 pour déboguer)

# Nombre de filtres à chaque niveau du U-Net (voir monai.networks.nets.BasicUNet).
UNET_FEATURES = (32, 32, 64, 128, 256, 32)

# Époques où l'on sauvegarde le modèle. Les premières donnent des sCT de mauvaise qualité,
# les dernières de bonne qualité : c'est ce qui donne un banc d'essai de qualité variée.
CHECKPOINT_EPOCHS = [1, 2, 5, 10, 20, 50, 100]

SEED = 42
