# Pipeline : banc d'essai des VLM juges

Ce dossier fait noter des images de CT synthétiques (sCT) par plusieurs VLM locaux servis par Ollama, et enregistre les scores dans un CSV.

Toutes les commandes se lancent **sur la station**, depuis ce dossier (`cd Synthesis/pipeline`), après un `uv sync` (voir le README de `Synthesis/`).

| Fichier | Rôle |
|---|---|
| `config.py` | adresse du serveur Ollama, liste des modèles, prompts, graine, réglages de l'analyse |
| `vlm_manager.py` | charger, décharger et lister les modèles (module et ligne de commande) |
| `judge.py` | faire noter une image par un VLM (score 0–100 et justification, JSON imposé) |
| `benchmark.py` | boucle complète : tous les modèles × toutes les images, vers un CSV |
| `analyze.py` | rattache les notes aux métriques et produit le leaderboard des juges |

## Gérer les modèles

```bash
uv run python vlm_manager.py list                     # modèles installés
uv run python vlm_manager.py ps                       # ce qui est en mémoire
uv run python vlm_manager.py load medgemma:latest     # charger en VRAM (affiche le temps et la VRAM)
uv run python vlm_manager.py unload medgemma:latest   # décharger
uv run python vlm_manager.py pull <modèle>            # télécharger (cache partagé : vérifier df -h / avant)
```

Le GPU est partagé : avant de lancer quoi que ce soit, `ps` permet de voir si quelqu'un d'autre l'utilise. On ne décharge jamais le modèle de quelqu'un d'autre.

## Tester un modèle sur une image

```bash
uv run python judge.py medgemma:latest chemin/vers/image.png
```

## Lancer le banc d'essai

Le banc d'essai est le `testbench.csv` produit par `sct_generation/make_testbench.py`. Seules ses deux premières colonnes, `item_id` et `image_path`, sont lues ici. On commence petit, sur val, avec le modèle le plus rapide :

```bash
uv run python benchmark.py --testbench ../sct_generation/work/testbench/val/gray/testbench.csv --models medgemma:latest
```

puis on lance tous les modèles de `config.py`, de préférence aux heures creuses :

```bash
uv run python benchmark.py --testbench ../sct_generation/work/testbench/val/gray/testbench.csv            # modèles déjà installés
uv run python benchmark.py --testbench ../sct_generation/work/testbench/val/gray/testbench.csv --pull     # télécharge aussi les manquants
```

Pour chaque modèle, le benchmark le télécharge si besoin (avec `--pull`), le charge en VRAM, lui fait noter toutes les images, puis le décharge avant de passer au suivant. Sans `--pull`, un modèle absent du cache est sauté avec un message. Un modèle qui ne se télécharge pas ou ne se charge pas est signalé et le benchmark passe au suivant. Pour ajouter un nouveau modèle au benchmark, il suffit de l'ajouter à `MODELS` dans `config.py` (ou de le passer avec `--models`).

Le banc d'essai de test ne se lance qu'une fois, à la fin, quand prompt et rendu ont été choisis sur val.

Les résultats sont écrits dans `Synthesis/results/judgements.csv`, une ligne par jugement :

| Colonne | Contenu |
|---|---|
| `model`, `prompt_name`, `item_id` | qui a jugé quoi, avec quel prompt |
| `score`, `reason` | réponse du VLM |
| `latency_s` | temps de réponse pour cette image |
| `vram_gb` | VRAM occupée par le modèle |
| `error` | vide si tout s'est bien passé |
| `timestamp` | date et heure du jugement |

Si le script est interrompu, il suffit de le relancer avec la même commande : les jugements déjà faits sont sautés. Les réponses sont plafonnées à `NUM_PREDICT` tokens : un modèle qui se met à boucler est noté en erreur au lieu de bloquer.

## Analyser les notes

```bash
uv run python analyze.py --testbench ../sct_generation/work/testbench/val/gray/testbench.csv
```

Le script rattache chaque note aux métriques de l'image grâce à `item_id` (seules les notes de ce banc d'essai sont prises), puis écrit dans `results/analysis/<split>_<style>/` :

| Fichier | Contenu |
|---|---|
| `leaderboard.csv` | une ligne par (modèle, prompt), du meilleur juge au moins bon |
| `correlations.csv` | Spearman entre la note et chaque métrique (coupe et volume) |
| `filtering.csv` | gain d'un filtrage par le juge, comparé à un filtrage oracle et au hasard |
| `generators.csv` | qualité de chaque générateur de sCT, dont la MAE par centre |
| `scores_and_metrics.csv` | chaque note avec les métriques de l'image notée |
| `scatter.png` | note du VLM en fonction de la MAE de la coupe, un graphique par modèle |

Les corrélations sont **orientées** : +1 veut toujours dire « le VLM donne les meilleures notes aux meilleurs sCT », 0 « aucun lien ». Pour la MAE, où plus bas = meilleur, on change donc le signe. Les colonnes principales du leaderboard :

| Colonne | Question |
|---|---|
| `spearman_mae_hu_slice` (+ `_ci_low`, `_ci_high`) | le classement des notes suit-il celui de la MAE ? Intervalle de confiance à 95 % par bootstrap sur les patients |
| `pearson_mae_hu_slice` | relation linéaire entre note et MAE |
| `spearman_intra` | à anatomie égale (même patient, même coupe), le VLM classe-t-il bien les sCT des différents générateurs ? La mesure la plus stricte |
| `auroc` | le VLM sépare-t-il les coupes acceptables (MAE < `ACCEPTABLE_THRESHOLD`) des autres ? 0,5 = hasard |
| `score_std` | dispersion des notes : proche de 0, le VLM donne toujours la même note |
| `latency_s_mean`, `vram_gb` | coût du juge |

La métrique de référence, le seuil d'acceptabilité, le nombre de tirages du bootstrap et la part gardée au filtrage se règlent dans la section « Analyse » de `config.py`.

## Règle importante

Le VLM ne voit que le PNG et le prompt. Les métriques (MAE, SSIM, gamma, DVH) ne sont jamais lues par `benchmark.py`. On les rattache aux scores au moment de l'analyse, grâce à `item_id`.
