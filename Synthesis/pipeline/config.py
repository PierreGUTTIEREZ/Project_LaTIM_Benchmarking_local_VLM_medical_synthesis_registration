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

# Nombre maximal de tokens générés par réponse. Une réponse normale en fait moins de 200 ;
# le plafond évite qu'un petit modèle qui se met à répéter la même phrase (ça arrive à
# température 0) bloque une image pendant des minutes. Une réponse coupée est retentée (voir plus bas).
NUM_PREDICT = 2048

# Les modèles qui « réfléchissent » avant de répondre (capacité « thinking » dans Ollama, par
# exemple Qwen3-VL 8B) écrivent d'abord un raisonnement caché, compté dans le plafond : avec
# 2048 tokens, 24 % des réponses de Qwen3-VL 8B étaient coupées avant même la note. Ces modèles
# ont donc un plafond plus haut. Les réponses déjà réussies n'en sont pas changées : à
# température 0, le plafond ne joue que sur les réponses qui l'atteignent.
NUM_PREDICT_THINKING = 8192

# Plafond propre à un modèle, si besoin (prioritaire sur les deux valeurs ci-dessus),
# par exemple {"minicpm-v:8b": 1024}.
NUM_PREDICT_BY_MODEL = {}

# --- Nouveaux essais après une erreur ---------------------------------------

# Deux pannes reviennent avec certains modèles :
#   - « réponse coupée après N tokens » : le modèle boucle, ou réfléchit trop longtemps ;
#   - « token repeat limit reached (status code: 500) » : Ollama arrête lui-même un modèle qui
#     répète le même token en boucle (souvent des espaces ou des retours à la ligne dans le JSON).
# Une image en erreur est retentée, jusqu'à MAX_ATTEMPTS essais au total. Le premier essai suit
# toujours le protocole (température 0, consigne seule). Les suivants ajoutent ce que décrit
# RETRY_STRATEGIES, dans l'ordre. Le numéro de l'essai réussi est écrit dans le CSV (colonne
# attempt) : l'analyse compte ces images à part, et on peut les exclure si besoin.
MAX_ATTEMPTS = 3
JSON_HINT = (
    'Answer only with a JSON object of the form {"score": <integer from 0 to 100>, '
    '"reason": "<one or two sentences>"}.'
)
RETRY_STRATEGIES = [
    # 2e essai : même réglage, plus un rappel du format attendu, en message système.
    {"name": "json_hint", "system": JSON_HINT, "options": {}},
    # 3e essai : en plus, un peu d'aléa et une pénalité de répétition pour sortir de la boucle,
    # et pas de réflexion si le modèle le permet.
    {"name": "json_hint_sampling", "system": JSON_HINT, "think": False,
     "options": {"temperature": 0.7, "repeat_penalty": 1.3, "seed": SEED + 1}},
]

# Pause avant un nouvel essai après une erreur du serveur (Ollama relance parfois le modèle).
RETRY_PAUSE_S = 2

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
    # Barème explicite par tranches, et points à regarder propres au crâne : teste si
    # des repères chiffrés rendent les notes plus cohérentes d'une image à l'autre.
    "brain_rubric_v1": (
        "This image is an axial slice of a head CT scan. It may have been generated "
        "by an AI model. Rate its image quality with this scale: "
        "0-20 unusable (anatomy unrecognizable or severe artifacts); "
        "21-40 poor (major blur or distortion, skull or brain badly rendered); "
        "41-60 fair (anatomy recognizable but clearly blurred or noisy); "
        "61-80 good (only minor imperfections); "
        "81-100 excellent (indistinguishable from a real clinical CT). "
        "Look in particular at the skull (continuous cortical bone with a realistic "
        "bright density), the brain (homogeneous soft-tissue density, plausible "
        "ventricles and sulci), air in the sinuses and around the head, and any "
        "artifact. Give the score and a short justification."
    ),
    # Démarche inverse : partir de 100 et retirer des points par défaut trouvé.
    # Teste si chercher les défauts discrimine mieux que juger l'ensemble.
    "defects_v1": (
        "This image is an axial slice of a CT scan that may have been generated by "
        "an AI model. Inspect it for defects: blur, loss of fine detail, unrealistic "
        "noise or texture, checkerboard or stripe patterns, wrong densities (bone "
        "too dark, soft tissue too bright), missing, duplicated or distorted "
        "structures, and implausible boundaries. Start from 100 and subtract points "
        "for each defect according to its severity. Give the final score from 0 to "
        "100 and list the defects you found as the justification."
    ),
}
DEFAULT_PROMPT = "realism_v1"

# --- Fichiers ---------------------------------------------------------------

# Le banc d'essai est produit par sct_generation/make_testbench.py, par exemple
# ../sct_generation/work/testbench/val/gray/testbench.csv. On le passe toujours
# explicitement avec --testbench : ainsi on ne lance jamais le test par mégarde
# (le test ne sert qu'à la fin, tous les réglages se font sur val).
TESTBENCH_CSV = None

RESULTS_DIR = SYNTHESIS_DIR / "results"
DEFAULT_RESULTS_CSV = RESULTS_DIR / "judgements.csv"

# Version exacte de chaque modèle utilisé (empreinte, taille, quantification, capacités), mise
# à jour à chaque benchmark : les tags comme « latest » changent de contenu avec le temps.
MODELS_CSV = RESULTS_DIR / "ollama_models.csv"

# --- Analyse (analyze.py) -----------------------------------------------------

ANALYSIS_DIR = RESULTS_DIR / "analysis"

# Métrique de référence du leaderboard. On prend celle de la coupe rendue, puisque c'est
# ce que le VLM voit (les colonnes du volume entier sont aussi analysées dans correlations.csv).
TARGET_METRIC = "mae_hu_slice"

# Métriques où « plus bas = meilleur ». On change le signe de leurs corrélations pour que,
# dans tous les tableaux, une valeur proche de +1 veuille dire « le VLM donne les meilleures
# notes aux meilleurs sCT », et une valeur proche de 0 « le VLM note au hasard ».
# Les écarts de dose (sct_generation/dose.py) en font partie ; le taux gamma, non.
LOWER_IS_BETTER = {"mae_hu", "mae_hu_slice"} | {
    f"{metric}_{modality}"
    for metric in ("dose_diff", "ptv_d98", "ptv_d2", "body_dmean")
    for modality in ("photons", "protons")
}

# AUROC : un sCT est « acceptable » si sa MAE sur la coupe est sous ce seuil (en HU).
# L'AUROC mesure si le VLM sépare les acceptables des autres (1 = parfait, 0,5 = hasard).
ACCEPTABLE_METRIC = "mae_hu_slice"
ACCEPTABLE_THRESHOLD = 100

# Le seuil de 100 HU est un choix : l'AUROC est aussi calculée avec ces seuils, pour vérifier
# que le classement des juges n'en dépend pas (colonnes auroc_80, auroc_120 du leaderboard).
ACCEPTABLE_THRESHOLDS_SENSITIVITY = (80, 120)

# Intervalles de confiance à 95 % par bootstrap sur les patients : on retire des patients
# au hasard (avec remise) et on recalcule la corrélation, BOOTSTRAP_SAMPLES fois.
BOOTSTRAP_SAMPLES = 1000

# Filtrage : part des images gardées quand on ne garde que les mieux notées.
KEEP_FRACTION = 0.5
