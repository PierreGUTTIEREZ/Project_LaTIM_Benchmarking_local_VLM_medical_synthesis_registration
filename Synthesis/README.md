# Volet Synthèse

| Dossier | Rôle |
|---|---|
| [`sct_generation/`](sct_generation/) | entraîner un modèle IRM → CT et produire des sCT de qualité variée |
| [`pipeline/`](pipeline/) | faire noter les sCT (rendus en PNG) par les VLM via Ollama |
| [`results/`](results/) | CSV des jugements des VLM |

## Où lancer le code

Tout se lance **sur la station `latim5090`**, en SSH via Tailscale (`ssh student@100.106.248.35`) :
c'est là que se trouvent le GPU, les données (`/home/student/DATA/SYNTHESIS/brain`) et Ollama
(qui n'accepte que les connexions locales).

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
bibliothèque : `uv add <nom>` (et non `pip install`, qui ne serait pas enregistré).

## Ordre des étapes

1. Découper SynthRAD2023 en train / val / test **par patient**.
2. Entraîner le modèle de synthèse sur train, régler sur val, en sauvegardant des checkpoints à plusieurs époques.
3. Prédire les sCT des patients de test avec chaque checkpoint : on obtient des sCT allant du mauvais au bon.
4. Calculer les métriques (MAE, PSNR, SSIM, gamma, DVH) contre le vrai CT et rendre les sCT en PNG.
5. Lancer le banc d'essai des VLM sur ces PNG, puis corréler les scores avec les métriques.
