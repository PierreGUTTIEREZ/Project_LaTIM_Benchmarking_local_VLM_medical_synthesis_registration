#!/usr/bin/env bash
# Lance toute la chaîne qui suit l'entraînement du U-Net, pour une partie du split :
# sCT du U-Net et des modèles pré-entraînés, banc d'essai, benchmark des VLM, analyse.
#
#   bash run_all.sh val              # mise au point du juge : styles gray et rgb3
#   bash run_all.sh val gray         # un seul style
#   bash run_all.sh test gray        # résultats finaux, avec le style choisi sur val
#
# Réglages facultatifs, à écrire devant la commande :
#   RUN=unet_baseline          run du U-Net dont on prend les checkpoints (défaut : unet_baseline)
#   MAX_PATIENTS=2             limiter le nombre de patients (essai rapide)
#   MODELS="medgemma:latest"   limiter les VLM (défaut : ceux de pipeline/config.py déjà installés)
#   PROMPTS="realism_v1 brain_rubric_v1 defects_v1"
#                              prompts à tester (défaut : DEFAULT_PROMPT de pipeline/config.py)
#   DOSE=1                     calculer aussi les métriques de dose (dose.py, long, sur le processeur)
#
# Relancer la même commande retente les images restées en erreur (réponse coupée, erreur 500
# « token repeat limit », justification vide), avec les nouveaux essais de pipeline/config.py.
#
# Comparer les trois prompts sur val (chaque prompt multiplie la durée du benchmark) :
#   PROMPTS="realism_v1 brain_rubric_v1 defects_v1" bash run_all.sh val gray
#
# Essai complet rapide, à faire avant les longs calculs :
#   MAX_PATIENTS=2 bash run_all.sh val gray
#
# À lancer sur la station, depuis le dossier Synthesis/, dans tmux, une fois train.py terminé.
# Chaque étape saute ce qui est déjà fait : après une coupure, relancer la même commande.

set -euo pipefail

SPLIT=${1:-}
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "Usage : bash run_all.sh val [styles...]   ou   bash run_all.sh test <style>"
    exit 1
fi
shift
STYLES=("$@")
if [[ ${#STYLES[@]} -eq 0 ]]; then
    if [[ "$SPLIT" == "test" ]]; then
        echo "Pour le test, préciser le style de rendu choisi sur val, par exemple : bash run_all.sh test gray"
        exit 1
    fi
    STYLES=(gray rgb3)
fi

RUN=${RUN:-unet_baseline}
LIMIT=()
if [[ -n "${MAX_PATIENTS:-}" ]]; then
    LIMIT=(--max-patients "$MAX_PATIENTS")
fi
VLMS=()
if [[ -n "${MODELS:-}" ]]; then
    read -r -a VLMS <<< "$MODELS"
    VLMS=(--models "${VLMS[@]}")
fi
read -r -a PROMPT_NAMES <<< "${PROMPTS:-}"

SYNTHESIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
step() { echo; echo "=== $(date '+%H:%M:%S')  $*"; }

echo "Partie : $SPLIT · styles : ${STYLES[*]} · run U-Net : $RUN · patients : ${MAX_PATIENTS:-tous} · VLM : ${MODELS:-installés} · prompts : ${PROMPTS:-par défaut} · dose : ${DOSE:-0}"
echo "GPU partagé : vérifier que personne d'autre ne s'en sert (nvidia-smi, ollama ps)."

cd "$SYNTHESIS_DIR/sct_generation"

step "1/4  sCT du U-Net (predict.py)"
uv run python predict.py --run "$RUN" --split "$SPLIT"

step "2/4  sCT des modèles pré-entraînés Li, Fu, Spadea (pretrained.py)"
# Un échec ici n'arrête pas la chaîne : le banc d'essai se fera avec les sCT disponibles.
uv run python pretrained.py --split "$SPLIT" "${LIMIT[@]}" \
    || echo "Attention : pretrained.py a signalé des échecs (voir plus haut). On continue avec les sCT disponibles."

step "3/4  Banc d'essai : métriques et PNG (make_testbench.py)"
for style in "${STYLES[@]}"; do
    uv run python make_testbench.py --split "$SPLIT" --style "$style" "${LIMIT[@]}"
done

if [[ "${DOSE:-0}" == "1" ]]; then
    step "3b   Métriques de dose : gamma, DVH (dose.py)"
    # Un échec ici n'arrête pas la chaîne : l'analyse se fera sans les patients manquants.
    uv run python dose.py --split "$SPLIT" "${LIMIT[@]}" \
        || echo "Attention : dose.py a signalé des échecs (voir plus haut). On continue."
fi

cd "$SYNTHESIS_DIR/pipeline"
step "     Versions des modèles Ollama (results/ollama_models.csv)"
uv run python vlm_manager.py info || echo "Attention : impossible de lire les versions des modèles. On continue."
for style in "${STYLES[@]}"; do
    testbench="../sct_generation/work/testbench/$SPLIT/$style/testbench.csv"

    if [[ ${#PROMPT_NAMES[@]} -eq 0 ]]; then
        step "4/4  Benchmark des VLM, style $style (benchmark.py)"
        uv run python benchmark.py --testbench "$testbench" "${VLMS[@]}"
    fi
    for prompt in "${PROMPT_NAMES[@]}"; do
        step "4/4  Benchmark des VLM, style $style, prompt $prompt (benchmark.py)"
        uv run python benchmark.py --testbench "$testbench" --prompt "$prompt" "${VLMS[@]}"
    done

    step "     Analyse, style $style (analyze.py)"
    uv run python analyze.py --testbench "$testbench"
done

step "Terminé. Résultats dans $SYNTHESIS_DIR/results/analysis/"
