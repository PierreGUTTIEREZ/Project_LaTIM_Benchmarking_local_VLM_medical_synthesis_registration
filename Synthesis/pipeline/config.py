"""Paramètres du banc d'essai.

Tout ce qu'on est amené à modifier d'une expérience à l'autre est regroupé ici,
pour ne pas avoir à fouiller dans le code.
"""

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
SYNTHESIS_DIR = PIPELINE_DIR.parent

# --- Serveur Ollama ---------------------------------------------------------

# Sur latim5090, Ollama est un service partagé qui n'écoute qu'en local :
# le pipeline doit donc être lancé sur la station (via SSH), pas depuis un portable.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Le premier chargement d'un gros modèle (30b) peut prendre plusieurs minutes.
REQUEST_TIMEOUT_S = 600

# --- Modèles à évaluer ------------------------------------------------------

# Du plus petit au plus gros : on valide la chaîne avec un petit modèle
# avant de lancer les gros. On peut ajouter ou retirer des modèles librement.
#
# Ceux marqués « installé » sont déjà dans le cache Ollama de latim5090. Les autres
# (ceux du vademecum) sont téléchargés automatiquement avec `benchmark.py --pull`,
# ou à la main avec `vlm_manager.py pull <modèle>`. Le cache est partagé :
# vérifier la place disque (df -h /) avant de télécharger.
MODELS = [
    "moondream",         # ~1,8B, ultra-rapide : test de bout en bout
    "medgemma:latest",   # 4B, médical (installé ; c'est le medgemma:4b du vademecum)
    "llava:7b",          # 7B, baseline généraliste
    "gemma4:latest",     # 8B, généraliste (installé)
    "qwen3-vl:8b",       # 8B, généraliste récent
    "minicpm-v:8b",      # 8B, bon rapport qualité/taille
    "llava:13b",         # 13B
    "medgemma:27b",      # 27B, médical (installé)
    "qwen3.8:27b",       # 27B, généraliste (installé)
    "qwen3-vl:30b",      # 30B, le plus gros qui tienne en VRAM
]

# --- Reproductibilité -------------------------------------------------------

TEMPERATURE = 0
SEED = 42

# --- Prompts ----------------------------------------------------------------

# On garde chaque formulation sous un nom, et ce nom est écrit dans le CSV
# de résultats. Pour tester une autre formulation (biais de prompt), on ajoute
# une entrée ici plutôt que de modifier une formulation déjà utilisée.
# Règle : ne jamais mettre de métrique ni d'indice sur la vérité terrain dans un prompt.
PROMPTS = {
    "realism_v1": (
        "This image is an axial slice of a CT scan. It may have been generated "
        "by an AI model from an MRI or a CBCT. Assess how anatomically realistic "
        "it is: plausible anatomy, sharp boundaries between tissues, realistic "
        "contrast between bone, soft tissue and air, and absence of artifacts "
        "(blur, unusual noise, checkerboard patterns, missing or distorted structures). "
        "Give a score from 0 (clearly unrealistic) to 100 (indistinguishable from "
        "a real CT), and a short justification."
    ),
}
DEFAULT_PROMPT = "realism_v1"

# --- Fichiers ---------------------------------------------------------------

# Le CSV du banc d'essai sera fourni plus tard avec les données
# (voir le format attendu dans benchmark.py). En attendant, on le passe
# en argument avec --testbench.
TESTBENCH_CSV = None

RESULTS_DIR = SYNTHESIS_DIR / "results"
DEFAULT_RESULTS_CSV = RESULTS_DIR / "judgements.csv"
