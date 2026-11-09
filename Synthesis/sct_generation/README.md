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
| `dose.py` | métriques de dose (gamma, DVH) : plan optimisé sur le vrai CT, recalculé sur chaque sCT |

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

**8. (Facultatif) Calculer les métriques de dose**

```bash
uv run python dose.py --split val --max-patients 2   # essai
uv run python dose.py --split val                    # puis --split test à la fin
```

Même principe que SynthRAD2023 : pour chaque patient, un plan est optimisé sur le **vrai CT**, puis recalculé tel quel (mêmes faisceaux, mêmes poids) sur chaque sCT, et les deux doses sont comparées. Le calcul de dose est fait par [pyRadPlan](https://github.com/e0404/pyRadPlan) (portage Python de matRad), en photons (IMRT 5 faisceaux, 6 MV) et en protons (2 faisceaux latéraux), sur le processeur : il peut tourner pendant le benchmark des VLM. Compter quelques minutes par patient.

| Métrique (par modalité) | Contenu | Meilleur si |
|---|---|---|
| `gamma_photons`, `gamma_protons` | taux de réussite gamma 2 %/2 mm (seuil 10 %), en % | plus haut |
| `dose_diff_…` | écart de dose moyen là où la dose dépasse 10 % du maximum, en % de la prescription | plus bas |
| `ptv_d98_…`, `ptv_d2_…` | écart de D98 (couverture) et de D2 (point chaud) de la cible, en % de la prescription | plus bas |
| `body_dmean_…` | écart de dose moyenne dans le corps, en % de la prescription | plus bas |

Résultat : `work/testbench/<split>/dose_metrics.csv`, que `pipeline/analyze.py` lit tout seul. **Pas besoin de refaire le benchmark** : les notes des VLM ne changent pas, il suffit de relancer `analyze.py`. Les doses de référence (plan sur le vrai CT) sont dans `work/dose/<split>/<patient>/` pour vérification dans 3D Slicer.

> **Limite à dire dans le rapport :** SynthRAD2023 ne fournit ni contours ni plans cliniques. La cible est donc une sphère de 15 mm de rayon, centrée sur les tissus mous de la tête, avec une prescription de 60 Gy (`DOSE_*` dans `config.py`). Les métriques mesurent l'effet des erreurs du sCT sur un plan réaliste, pas sur le plan réel du patient.

## Données patient

Tout ce qui est dérivé des images (volumes pré-traités, checkpoints, sCT, PNG) est écrit dans `work/`, qui est exclu de git.
