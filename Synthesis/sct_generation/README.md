# Génération des sCT (IRM → CT)

Ce dossier entraîne un modèle de synthèse IRM → CT et produit les CT synthétiques (sCT) qui seront ensuite jugés par les VLM.

Le modèle est un **U-Net 2D** (MONAI) : il prend une coupe axiale d'IRM et prédit la coupe CT correspondante. Il est entraîné avec une perte L1. C'est un choix volontairement simple, rapide à entraîner, qui suffit à produire des sCT de qualité variée.

| Fichier | Rôle |
|---|---|
| `config.py` | chemins, plage de HU, hyperparamètres, époques des checkpoints |
| `make_split.py` | découpage train / val / test par patient |
| `preprocess.py` | normalise les volumes une fois pour toutes et les enregistre en `.npy` |
| `data.py` | normalisation, lecture des volumes pré-traités, tirage des coupes d'entraînement |
| `model.py` | construction du U-Net et prédiction d'un volume entier |
| `train.py` | entraînement, checkpoints, MAE de validation |
| `predict.py` | génère les sCT (NIfTI, en HU) avec chaque checkpoint |

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
| train | entraîner le modèle de synthèse | non |
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

Les checkpoints sont sauvegardés aux époques de `CHECKPOINT_EPOCHS` (1, 2, 5, 10, 20, 50, 100). Les premiers donnent des sCT flous et peu réalistes, les derniers des sCT de bonne qualité : c'est ce qui donne un banc d'essai de qualité variée. La MAE de validation de chaque checkpoint est écrite dans `work/runs/unet_baseline/training_log.csv`.

**5. Générer les sCT**

```bash
uv run python predict.py --run unet_baseline              # patients de test, tous les checkpoints
uv run python predict.py --run unet_baseline --split val  # patients de val, pour mettre au point le juge
```

Résultat : `work/runs/unet_baseline/predictions/<split>/epoch_XXX/<patient>/sct.nii.gz`.

## Données patient

Tout ce qui est dérivé des images (volumes pré-traités, checkpoints, sCT) est écrit dans `work/`, qui est exclu de git.
