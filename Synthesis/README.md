# Volet Synthèse

| Dossier | Rôle |
|---|---|
| [`sct_generation/`](sct_generation/) | produire des sCT de qualité variée (U-Net entraîné par nos soins + 3 modèles pré-entraînés), puis le banc d'essai : métriques et rendu PNG |
| [`pipeline/`](pipeline/) | faire noter les PNG par les VLM via Ollama, puis analyser les notes (leaderboard) |
| [`results/`](results/) | CSV des jugements des VLM et résultats de l'analyse |

Le guide complet, avec le tutoriel commande par commande, est dans [`Mode d'emploi du volet Synthèse.html`](Mode%20d'emploi%20du%20volet%20Synth%C3%A8se.html).

## Où lancer le code

Tout se lance **sur la station `latim5090`**, en SSH via Tailscale (`ssh student@100.106.248.35`) :
c'est là que se trouvent le GPU, les données (`/home/student/DATA/SYNTHESIS/brain`), Ollama
(qui n'accepte que les connexions locales) et les modèles pré-entraînés installés par Vincent
(`/home/student/SYNTHETIC_CT/`).

Le compte `student` est partagé par toutes les équipes : même dossier personnel, même GPU,
même cache de modèles Ollama. On ne décharge ni ne supprime jamais un modèle qu'on n'a pas
chargé soi-même, et on lance les longs calculs aux heures creuses.

## Environnement Python

L'environnement est géré par [`uv`](https://docs.astral.sh/uv/), comme recommandé sur la station.
`pyproject.toml` liste les dépendances et `uv.lock` fige leurs versions exactes, y compris le
PyTorch compilé pour CUDA 13 dont la RTX 5090 a besoin.

```bash
cd Synthesis
uv sync                                                                  # une seule fois
uv run python -c "import torch; print(torch.cuda.is_available())"       # doit afficher True
```

Ensuite, chaque script se lance avec `uv run python ...`, depuis son dossier. Pour ajouter une
bibliothèque : `uv add <nom>` sur ton PC (et non `pip install`, qui ne serait pas enregistré),
puis commiter `pyproject.toml` **et** `uv.lock`. Si `pyproject.toml` a changé sans que `uv.lock`
suive, lancer `uv lock` sur ton PC avant de commiter : sinon `uv` modifie `uv.lock` sur la
station et le prochain `git pull` y sera bloqué.

## Ordre des étapes

1. Découper SynthRAD2023 en train / val / test **par patient** (`make_split.py`).
2. Entraîner le U-Net sur train, régler sur val, en sauvegardant des checkpoints à plusieurs époques (`preprocess.py`, `train.py`).
3. Prédire les sCT avec chaque checkpoint (`predict.py`) et avec les trois modèles pré-entraînés Li, Fu, Spadea (`pretrained.py`) : on obtient des sCT allant du mauvais au bon.
4. Calculer les métriques (MAE, PSNR, SSIM) contre le vrai CT et rendre les coupes en PNG (`make_testbench.py`).
5. Faire noter les PNG par les VLM (`benchmark.py`), d'abord sur val pour régler prompt et rendu, puis une seule fois sur test.
6. Corréler les notes avec les métriques et produire le leaderboard (`analyze.py`).

Une fois le U-Net entraîné, `run_all.sh` enchaîne les étapes 3 à 6 en une commande, sur la station, dans `tmux` :

```bash
MAX_PATIENTS=2 bash run_all.sh val gray   # essai complet rapide sur deux patients de val
bash run_all.sh val                       # mise au point du juge sur val (styles gray et rgb3)
bash run_all.sh test gray                 # résultats finaux, avec le style choisi sur val
```

Les métriques de dose (gamma, DVH) demandent un moteur de calcul de dose : elles ne sont pas encore calculées.
