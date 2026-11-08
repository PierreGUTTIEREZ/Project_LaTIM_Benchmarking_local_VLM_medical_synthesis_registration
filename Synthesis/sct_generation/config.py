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

# --- Modèles pré-entraînés : FedSynthCT-Brain (Li, Fu, Spadea) -----------------

# Vincent a installé sur la station trois modèles IRM T1 -> CT déjà entraînés, issus de
# SlicerModalityConverter (https://github.com/ciroraggio/SlicerModalityConverter), avec un
# script pour les lancer (guide : /home/student/SYNTHETIC_CT/GUIDE.md). Ils tournent dans
# son environnement Python à lui : pretrained.py se contente d'appeler son script.
PRETRAINED_PYTHON = Path("/home/vincent/venvs/TS/bin/python")
PRETRAINED_SCRIPT = Path("/home/student/SYNTHETIC_CT/code/synthesize.py")
PRETRAINED_DEVICE = "cuda:0"

# Suffixes des fichiers écrits par son script : <patient>_li.synct.nii.gz, etc.
PRETRAINED_MODELS = ["li", "fu", "spadea"]
PRETRAINED_DIR = WORK_DIR / "pretrained"

# Ces modèles ont été entraînés en partie sur SynthRAD2023 (Raggio et al., 2025) : le centre C
# a servi à l'entraînement (29 patients, on ne sait pas lesquels), le centre B seulement à leur
# test, le centre A jamais. Sur nos patients du centre C, ils ont donc peut-être déjà vu le
# patient : leurs sCT y sont sans doute meilleurs que sur un patient vraiment nouveau.
# make_testbench.py le note dans la colonne maybe_seen_in_training.
PRETRAINED_SEEN_CENTERS = ["C"]

# --- Banc d'essai : coupes, rendu PNG, métriques -------------------------------

TESTBENCH_DIR = WORK_DIR / "testbench"

# Coupes axiales rendues pour chaque patient, en position relative dans l'étendue du corps
# (0 = première coupe qui contient le patient, en bas ; 1 = dernière, en haut). Ce sont les
# mêmes coupes pour tous les sCT d'un patient : on peut comparer les modèles à anatomie égale.
SLICE_POSITIONS = [0.35, 0.5, 0.65]

# Marge autour du corps quand on recadre la coupe, en pixels.
CROP_MARGIN = 10

# Styles de rendu. `windows` est une liste de fenêtres (HU min, HU max) : une seule fenêtre
# donne une image en niveaux de gris, trois fenêtres une image couleur (rouge, vert, bleu).
# `size` est la taille du PNG carré, en pixels.
RENDER_STYLES = {
    # Fenêtre large : air, tissus mous et os visibles sur la même image.
    "gray": {"windows": [(-1000, 1000)], "size": 512},
    # Rendu recommandé par Google pour MedGemma : rouge = fenêtre large, vert = tissus mous,
    # bleu = cerveau, en 896 × 896.
    "rgb3": {"windows": [(-1024, 1024), (-135, 215), (0, 80)], "size": 896},
}
DEFAULT_RENDER_STYLE = "gray"

# Avant le rendu, tout ce qui est hors du masque du corps est mis à l'air (-1024 HU), pour tous
# les sCT. Sans ça, le VLM pourrait reconnaître un modèle à son fond (bruité pour le U-Net,
# parfaitement noir pour les modèles pré-entraînés) au lieu de juger l'anatomie.
# Les métriques, elles, sont toujours calculées dans le masque.
MASK_OUTSIDE_BODY = True
