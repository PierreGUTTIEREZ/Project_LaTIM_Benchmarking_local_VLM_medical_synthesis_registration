# Pipeline : banc d'essai des VLM juges

Ce dossier fait noter des images de CT synthétiques (sCT) par plusieurs VLM locaux servis par Ollama, et enregistre les scores dans un CSV.

Toutes les commandes se lancent **sur la station**, depuis ce dossier (`cd Synthesis/pipeline`), après un `uv sync` (voir le README de `Synthesis/`).

| Fichier | Rôle |
|---|---|
| `config.py` | adresse du serveur Ollama, liste des modèles, prompts, graine |
| `vlm_manager.py` | charger, décharger et lister les modèles (module et ligne de commande) |
| `judge.py` | faire noter une image par un VLM (score 0–100 et justification, JSON imposé) |
| `benchmark.py` | boucle complète : tous les modèles × toutes les images, vers un CSV |

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

Le banc d'essai est un CSV avec deux colonnes obligatoires, `item_id` et `image_path` (voir `benchmark.py`). On commence petit, avec le modèle le plus rapide :

```bash
uv run python benchmark.py --testbench chemin/vers/testbench.csv --models medgemma:latest
```

puis on lance tous les modèles de `config.py`, de préférence aux heures creuses :

```bash
uv run python benchmark.py --testbench chemin/vers/testbench.csv            # modèles déjà installés
uv run python benchmark.py --testbench chemin/vers/testbench.csv --pull     # télécharge aussi les manquants
```

Pour chaque modèle, le benchmark le télécharge si besoin (avec `--pull`), le charge en VRAM, lui fait noter toutes les images, puis le décharge avant de passer au suivant. Sans `--pull`, un modèle absent du cache est sauté avec un message. Pour ajouter un nouveau modèle au benchmark, il suffit de l'ajouter à `MODELS` dans `config.py` (ou de le passer avec `--models`).

Les résultats sont écrits dans `Synthesis/results/judgements.csv`, une ligne par jugement :

| Colonne | Contenu |
|---|---|
| `model`, `prompt_name`, `item_id` | qui a jugé quoi, avec quel prompt |
| `score`, `reason` | réponse du VLM |
| `latency_s` | temps de réponse pour cette image |
| `vram_gb` | VRAM occupée par le modèle |
| `error` | vide si tout s'est bien passé |
| `timestamp` | date et heure du jugement |

Si le script est interrompu, il suffit de le relancer avec la même commande : les jugements déjà faits sont sautés.

## Règle importante

Le VLM ne voit que le PNG et le prompt. Les métriques (MAE, SSIM, gamma, DVH) ne sont jamais lues par `benchmark.py`. On les rattache aux scores au moment de l'analyse, grâce à `item_id`.
