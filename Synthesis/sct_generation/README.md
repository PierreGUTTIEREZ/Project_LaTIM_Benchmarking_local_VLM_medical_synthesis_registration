# Génération des sCT et du banc d'essai

Ce dossier produit les CT synthétiques (sCT) qui seront jugés par les VLM, puis le banc d'essai : les coupes rendues en PNG et leurs métriques contre le vrai CT.

Les sCT viennent de deux sources :

- **un U-Net 2D** (MONAI) que l'on entraîne nous-mêmes : il prend une coupe axiale d'IRM et prédit la coupe CT correspondante, avec une perte L1. En gardant des checkpoints à plusieurs époques, on obtient des sCT allant du très flou au bon ;
- **trois modèles pré-entraînés** (FedSynthCT-Brain : Li, Fu, Spadea), installés par Vincent sur la station depuis [SlicerModalityConverter](https://github.com/ciroraggio/SlicerModalityConverter). Ce sont des modèles publiés, d'un niveau proche de l'état de l'art.

| Fichier | Rôle |
|---|---|
| `config.py` | chemins, plage de HU, hyperparamètres, modèles pré-entraînés, coupes et styles de rendu |
| `make_split.py` | découpage train / val / test par patient |
| `preprocess.py` | normalise les volumes une fois pour toutes et les enregistre en `.npy` |
| `data.py` | normalisation, lecture des volumes pré-traités, tirage des coupes d'entraînement |
| `model.py` | construction du U-Net et prédiction d'un volume entier |
| `train.py` | entraînement, checkpoints, MAE de validation |
| `predict.py` | génère les sCT du U-Net (NIfTI, en HU) avec chaque checkpoint |
| `pretrained.py` | génère les sCT des modèles Li, Fu et Spadea (appelle le script de Vincent) |
| `metrics.py` | MAE, PSNR et SSIM dans le masque du corps (définitions de SynthRAD2023) |
| `render.py` | fenêtrage, recadrage et rendu d'une coupe en PNG |
| `make_testbench.py` | métriques + PNG de tous les sCT disponibles, écrit `testbench.csv` |

## Étapes

Toutes les commandes se lancent sur la station, depuis ce dossier, après un `uv sync` (voir le README de `Synthesis/`).

**1. Vérifier `SYNTHRAD_DIR`** dans `config.py` : `/home/student/DATA/SYNTHESIS/brain`, le dossier qui contient directement `1BA001/`, `1BA005/`…

**2. Découper le jeu de données**

```bash
uv run python make_split.py     # -> splits/task1_brain.csv
```

Le découpage se fait **par patient** : toutes les coupes d'un même patient vont dans la même partie, sinon le modèle verrait en test des coupes presque identiques à celles de l'entraînement. Il est aussi fait centre par centre, pour que chaque centre hospitalier soit présent dans train, val et test. Par défaut : 70 % train, 15 % val, 15 % test, graine 42. Le script refuse d'écraser un split existant. Les fichiers de `splits/` sont à commiter.

| Partie | Sert à | Utilisée par le VLM ? |
|---|---|---|
| train | entraîner le U-Net | non |
| val | choisir les hyperparamètres ; régler le prompt et le rendu PNG du juge | oui, pour la mise au point |
| test | produire les sCT du banc d'essai final | oui, pour les résultats rapportés |

**3. Pré-traiter les volumes**

```bash
uv run python preprocess.py
```

**4. Entraîner**

```bash
uv run python train.py --run unet_baseline
```

Le GPU est partagé : vérifier avec `nvidia-smi` que personne ne l'utilise intensivement, et lancer les longs entraînements aux heures creuses.

Les checkpoints sont sauvegardés aux époques de `CHECKPOINT_EPOCHS` (1, 2, 5, 10, 20, 50, 100). Les premiers donnent des sCT flous et peu réalistes, les derniers des sCT de bonne qualité. La MAE de validation de chaque checkpoint est écrite dans `work/runs/unet_baseline/training_log.csv`.

**5. Générer les sCT du U-Net**

```bash
uv run python predict.py --run unet_baseline              # patients de test, tous les checkpoints
uv run python predict.py --run unet_baseline --split val  # patients de val, pour mettre au point le juge
```

Résultat : `work/runs/unet_baseline/predictions/<split>/epoch_XXX/<patient>/sct.nii.gz`.

**6. Générer les sCT des modèles pré-entraînés**

```bash
uv run python pretrained.py --split val      # puis sans --split pour le test
```

Le script appelle, patient par patient, celui de Vincent (`/home/student/SYNTHETIC_CT/code/synthesize.py`, avec son Python `/home/vincent/venvs/TS/bin/python`). Résultat : `work/pretrained/<split>/<patient>/<patient>_{li,fu,spadea}.synct.nii.gz`, sur la grille de l'IRM, plus `figures/<patient>_montage.png` pour vérifier à l'œil. Les patients déjà faits sont sautés.

> **Attention :** FedSynthCT-Brain a été entraîné en partie sur SynthRAD2023. Le centre C a servi à son entraînement (29 patients, on ne sait pas lesquels), le centre B seulement à son test, le centre A jamais. Sur nos patients `1BC…`, ces modèles ont donc peut-être déjà vu le patient. Ce n'est pas gênant pour juger les VLM (chaque sCT a sa vraie métrique), mais il ne faut pas rapporter leur MAE sur le centre C comme une performance sur des patients nouveaux. La colonne `maybe_seen_in_training` de `testbench.csv` signale ces cas.

**7. Construire le banc d'essai**

```bash
uv run python make_testbench.py --split val                 # rendu « gray » par défaut
uv run python make_testbench.py --split val --style rgb3    # rendu couleur recommandé pour MedGemma
uv run python make_testbench.py                             # test, une fois les réglages fixés
```

Le script rassemble tous les sCT disponibles (chaque checkpoint de chaque run, et Li, Fu, Spadea), prend pour chaque patient les mêmes coupes axiales dans tous les sCT (`SLICE_POSITIONS`), met le fond hors du corps à l'air, recadre sur le corps et enregistre des PNG sans aucun texte. Il calcule aussi MAE, PSNR et SSIM sur le volume entier et sur la coupe rendue.

Résultat dans `work/testbench/<split>/<style>/` : `renders/*.png` et `testbench.csv`, que l'on passe ensuite à `pipeline/benchmark.py`. Avec 27 patients, 10 générateurs et 3 coupes, cela fait 810 images.

## Données patient

Tout ce qui est dérivé des images (volumes pré-traités, checkpoints, sCT, PNG) est écrit dans `work/`, qui est exclu de git.
