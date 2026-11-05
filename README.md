# Benchmarking de VLM locaux pour juger la synthèse et le recalage d'images médicales

> **Projet Commande Entreprise, IMT Atlantique × LaTIM (UMR 1101)**
> Sujet A : *Benchmarking de VLM locaux pour l'analyse de résultats de synthèse et de recalage d'images médicales par IA en cancérologie.*

![Python](https://img.shields.io/badge/python-3.11-blue)
![Ollama](https://img.shields.io/badge/VLM-Ollama%20(local)-black)
![Status](https://img.shields.io/badge/status-en%20cours-orange)

---

## Sommaire

1. [En une phrase](#en-une-phrase)
2. [Contexte](#contexte)
3. [Question de recherche](#question-de-recherche)
4. [Les deux volets du projet](#les-deux-volets-du-projet)
5. [Principe du juge VLM](#principe-du-juge-vlm)
6. [Données](#données)
7. [Modèles VLM évalués](#modèles-vlm-évalués)
8. [Métriques](#métriques)
9. [Installation](#installation)
10. [Démarrage rapide](#démarrage-rapide)
11. [Organisation du dépôt](#organisation-du-dépôt)
12. [Planning](#planning)
13. [Bonnes pratiques et règles du projet](#bonnes-pratiques-et-règles-du-projet)
14. [Confidentialité et éthique](#confidentialité-et-éthique)
15. [Références](#références)
16. [Équipe et encadrement](#équipe-et-encadrement)
17. [Glossaire](#glossaire)

---

## En une phrase

Nous cherchons à savoir si un **modèle vision-langage (VLM) exécuté localement** peut servir de **juge automatique** de la qualité d'une image synthétisée (pseudo-CT) ou d'un champ de recalage, **sans accès à la vérité terrain**, et quel modèle juge le mieux, à quel coût.

---

## Contexte

En radiothérapie, l'imagerie pilote tout le traitement :

| Modalité | Apport | Limites | Rôle en radiothérapie |
|---|---|---|---|
| **CT** | géométrie, densité électronique | faible contraste des tissus mous, irradiation | planification, calcul de dose |
| **IRM** | contraste des tissus mous (T1/T2/FLAIR) | pas de densité électronique, distorsions | délinéation, suivi |
| **CBCT** | imagerie embarquée au moment du traitement | artéfacts (scatter, cupping) | guidage, adaptation |

Ces modalités sont complémentaires. Elles posent deux problèmes d'IA :

- **La synthèse.** Le workflow *MRI-only* génère un **CT synthétique (sCT ou pseudo-CT)** à partir de l'IRM, ou à partir du CBCT, pour éviter un scanner supplémentaire. Le sCT doit être réaliste **et** dosimétriquement fidèle.
- **Le recalage.** Il aligne deux images, par exemple un poumon en inspiration et en expiration. La « bonne » déformation est **latente** : il n'existe pas de vérité terrain directe. Contrairement à la segmentation, les méthodes **conventionnelles** par optimisation restent souvent supérieures au deep learning. Nous les utilisons donc comme **oracle** pour produire des champs de référence.

Dans les deux cas, les annotations de référence sont rares ou absentes. On s'appuie sur des **surrogates** (TRE, Jacobien, gamma, DVH, etc.), et nous voulons savoir si un VLM peut compléter ou approcher ces indicateurs.

---

## Question de recherche

> Peut-on utiliser un **VLM local** comme *juge automatique* de la qualité d'une prédiction (image synthétisée, champ de déformation) **sans accès à la vérité terrain** ?
> Si oui, **quel modèle** juge le mieux, et **à quel coût** (temps, VRAM) ?

Hypothèses à tester :

- Un VLM **spécialisé médical** (`medgemma`) juge-t-il mieux qu'un VLM **généraliste plus gros** (`qwen3-vl:30b`) ?
- Un VLM qui juge bien une tâche jugera-t-il bien les autres (juge *transversal*) ?
- Le filtrage par le juge apporte-t-il un **gain mesurable en aval** ?

---

## Les deux volets du projet

### Volet 1 : Synthèse (IRM → CT, CBCT → CT)

**Objectif :** juger le **réalisme** du sCT et sa **pertinence clinique**, puis corréler ce jugement avec les métriques image et dosimétriques.

1. Générer ou récupérer des pseudo-CT de qualité variée : modèles à différents stades d'entraînement, cas d'échec volontaires.
2. Calculer les métriques image : **MAE, PSNR, SSIM**.
3. Calculer les métriques aval : **gamma analysis (2 %/2 mm)**, écarts de **DVH**, D<sub>max</sub>, D<sub>mean</sub>.
4. Faire juger chaque sCT par les VLM (score 0–100 et justification).
5. Corréler score VLM, métriques image et métriques de dose.

### Volet 2 : Recalage déformable du poumon

**Objectif :** filtrer les champs de déformation **réalistes** pour entraîner un modèle supervisé.

1. Recaler les paires inspiration/expiration avec une méthode **conventionnelle non supervisée** (par ex. variation totale, `itk-elastix`, ANTs). C'est l'**oracle de champs**.
2. Dégrader volontairement certains champs (variation de la régularisation λ, de l'échelle, ajout de bruit) pour couvrir toute la gamme de qualité.
3. Calculer les surrogates : **TRE** sur les points de repère, **déterminant du Jacobien** (pliage si négatif), **cohérence inverse**.
4. Rendre chaque résultat en PNG (grille de déformation, fusion avant/après) et faire juger son réalisme par les VLM.
5. Mesurer la corrélation entre score VLM et TRE/Jacobien.
6. **Filtrer** les champs jugés réalistes, entraîner un modèle supervisé dessus, puis évaluer la TRE obtenue. On compare à un filtrage **oracle** et à un filtrage **aléatoire**.

> ⚠️ Un champ peut *paraître* lisse et pourtant être faux. Le jugement du VLM est toujours croisé avec des indices numériques : Jacobien négatif, pliage, incohérence inverse.

---

## Principe du juge VLM

```
 ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌───────────────────┐
 │ Image +      │ →  │ Rendu PNG lisible│ →  │ VLM local    │ →  │ JSON {score,      │
 │ prédiction   │    │ (overlay, grille,│    │ (Ollama)     │    │        reason}    │
 │ (sCT, champ) │    │  fusion, légende)│    │ température 0│    └─────────┬─────────┘
 └──────────────┘    └──────────────────┘    └──────────────┘              │
                                                                           ▼
             Surrogate « vrai » (TRE, SSIM, gamma…) ──────────►  Spearman / Pearson / AUROC
             (JAMAIS montré au VLM)                               → leaderboard → filtrage
```

Points clés :

- **Notation à l'aveugle** : la vérité terrain et les surrogates ne sont jamais montrés au VLM, ni dans l'image ni dans le texte.
- **Sorties structurées obligatoires** : le schéma JSON est imposé avec le paramètre `format` d'Ollama (décodage contraint). Il ne suffit pas de « demander gentiment » du JSON, car les petits modèles s'en écartent.
- **Déterminisme** : `temperature = 0`.

Exemple minimal :

```python
import json, ollama

SCHEMA = {
    "type": "object",
    "properties": {
        "score":  {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}

resp = ollama.chat(
    model="medgemma:4b",
    messages=[{
        "role": "user",
        "content": "Rate the anatomical realism of this synthetic CT slice from 0 to 100.",
        "images": ["renders/sct_001.png"],
    }],
    format=SCHEMA,                  # JSON garanti
    options={"temperature": 0},     # déterministe
)
result = json.loads(resp["message"]["content"])
```

Boucle de benchmark (pseudo-code) :

```python
for model in VLM_MODELS:
    for item in testbench:                         # (image, prédiction, surrogate_vrai)
        png = render(item.image, item.prediction)
        rep = ollama.chat(model=model,
                          messages=[{"role": "user", "content": PROMPT, "images": [png]}],
                          format=JUDGEMENT_SCHEMA,
                          options={"temperature": 0})
        score = json.loads(rep["message"]["content"])["score"]
        log(model, item.id, item.surrogate, score, latency, vram)   # → CSV
# puis : spearman(score, surrogate), auroc, leaderboard, filtrage
```

---

## Données

| Jeu de données | Tâche | Contenu | Métriques associées |
|---|---|---|---|
| **SynthRAD2023** | Synthèse | 540 paires IRM–CT et CBCT–CT, pelvis et neuro | MAE, PSNR, SSIM, gamma, DVH |
| **DIR-Lab** | Recalage | 4D-CT pulmonaire, paires inspiration/expiration avec points de repère | TRE, Jacobien |
| **COPDgene** | Recalage | CT pulmonaires inspiration/expiration avec points de repère | TRE, Jacobien |
| **POPI** | Recalage | 4D-CT pulmonaire avec points de repère | TRE, Jacobien |

> Les données sont mises à disposition sur la station de calcul et **ne sont pas versionnées dans ce dépôt**. Il faut respecter les licences et conditions d'utilisation de chaque jeu de données (usage recherche, citation).

---

## Modèles VLM évalués

Ils sont exécutés en local via [Ollama](https://ollama.com) sur une **RTX 5090 (32 Go de VRAM)**. On commence par les petits modèles, puis on monte en taille.

| Modèle | Taille | Pourquoi le tester |
|---|---|---|
| `moondream` | ~1,8B | minuscule, ultra-rapide : test de bout en bout |
| `llava:7b` | 7B | baseline généraliste répandue |
| `llava:13b` | 13B | version plus capable |
| `medgemma:4b` | 4B | spécialisé imagerie médicale |
| `medgemma:27b` | 27B (q4) | version médicale plus puissante |
| `qwen3-vl:8b` | 8B | excellent VLM généraliste récent |
| `minicpm-v:8b` | 8B | bon rapport qualité/taille |
| `qwen3-vl:30b` | 30B (q4) | le plus capable qui tienne encore en VRAM |

---

## Métriques

### Pour évaluer les prédictions (surrogates)

| Tâche | Métrique | Signification |
|---|---|---|
| Synthèse | **MAE** (HU) | erreur absolue moyenne entre sCT et CT réel |
| Synthèse | **PSNR**, **SSIM** | fidélité du signal et similarité structurelle |
| Synthèse | **Gamma (2 %/2 mm)** | concordance dosimétrique entre plan sur CT et plan sur sCT |
| Synthèse | **DVH**, D<sub>max</sub>, D<sub>mean</sub> | écarts de dose sur les structures |
| Recalage | **TRE** (mm) | erreur résiduelle sur les points de repère après recalage |
| Recalage | **Déterminant du Jacobien** | plausibilité du champ (pliage si ≤ 0) |
| Recalage | **Cohérence inverse** | le champ aller-retour revient-il à l'identité ? |

### Pour évaluer le juge VLM

| Métrique | Question posée |
|---|---|
| **Spearman ρ** | le classement du VLM suit-il celui du vrai surrogate ? |
| **Pearson r** | relation linéaire entre score et surrogate |
| **AUROC** | le juge sépare-t-il les prédictions « acceptables » (par ex. TRE < 2 mm) des autres ? |
| **Coût** | temps par image et VRAM utilisée |

> Référence de comparaison : **SegAE** (ISBI 2026) atteint une corrélation de 0,902 avec le Dice en segmentation, pour 0,06 s par masque 3D.

---

## Installation

> Les calculs tournent sur la station distante (RTX 5090), accessible via **Tailscale**.

### 1. Environnement Python

```bash
conda create -n vlmlab python=3.11 -y
conda activate vlmlab

pip install SimpleITK nibabel monai itk-elastix \
            numpy scipy scikit-learn pandas matplotlib pillow \
            ollama pydantic

nvidia-smi    # vérifier que le GPU est visible
```

### 2. Ollama et les modèles

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull moondream      # test rapide de la chaîne complète
ollama pull medgemma:4b    # VLM spécialisé imagerie médicale
ollama pull qwen3-vl:8b    # VLM généraliste récent
ollama pull llava:7b       # baseline
```

### Antisèche Ollama

```bash
ollama list                          # modèles installés
ollama pull <modele>                 # télécharger un modèle
ollama run  <modele> "prompt" img.png  # test interactif
ollama ps                            # modèles chargés en mémoire
ollama stop <modele>                 # décharger de la VRAM
ollama rm   <modele>                 # supprimer du disque
```

---

## Démarrage rapide

Lire un volume médical et le préparer pour l'affichage :

```python
import SimpleITK as sitk
import numpy as np

img = sitk.ReadImage("case.nii.gz")                  # volume 3D
print(img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection())

arr = sitk.GetArrayFromImage(img)                    # numpy : (z, y, x)
arr = np.clip(arr, -1000, 1000)                      # fenêtrage CT (HU)
arr = (arr - arr.min()) / (np.ptp(arr) + 1e-8)       # normalisation [0, 1]
```

Tester un VLM sur une image :

```bash
ollama run moondream "Describe this image." ./renders/example.png
```

---

## Organisation du dépôt

> Structure **prévue**, à ajuster au fil du projet.

```
.
├── README.md
├── requirements.txt          # dépendances figées (reproductibilité)
├── configs/                  # paramètres des expériences (modèles, prompts, seuils)
├── notebooks/
│   └── 01_prise_en_main.ipynb  # chargement, rééchantillonnage, TRE, DVH
├── src/
│   ├── data/                 # chargement SynthRAD / DIR-Lab / COPDgene / POPI
│   ├── synthesis/            # génération sCT + métriques (MAE, PSNR, SSIM, gamma, DVH)
│   ├── registration/         # oracle conventionnel + TRE, Jacobien, cohérence inverse
│   ├── rendering/            # rendu PNG lisible par les VLM
│   ├── judge/                # prompts, schémas JSON, appels Ollama
│   └── analysis/             # corrélations, AUROC, leaderboard, figures
├── results/                  # CSV de logs et figures (sans données patient)
├── docs/                     # fiches de lecture, carnet de bord, rapport
└── data/                     # ⚠️ non versionné (.gitignore)
```

---

## Planning

| Semaine | Thème | Livrable principal |
|---|---|---|
| **S1** | Littérature et prise en main des données | Fiches de lecture, réponses aux questions de brainstorming, notebook de manipulation |
| **S2** | Banc d'essai, rendu et premier juge VLM | Prédictions de qualité variée, surrogates, PNG, premier modèle qui note |
| **S3** | Benchmark multi-modèles et corrélations | Leaderboard (Spearman / Pearson / AUROC, temps, VRAM) et figures |
| **S4** | Boucle de filtrage et restitution | Comparaison aux témoins (oracle, aléatoire), rapport, présentation |

**État d'avancement :**

- [ ] S1 : Fiches de lecture
- [ ] S1 : Questions de brainstorming
- [ ] S1 : Notebook de prise en main (volume, rééchantillonnage, TRE, DVH)
- [ ] S2 : Banc d'essai de qualité variée (synthèse et recalage)
- [ ] S2 : Rendus PNG et premier juge VLM
- [ ] S3 : Benchmark de tous les modèles et leaderboard
- [ ] S4 : Filtrage des champs, réentraînement, évaluation de la TRE
- [ ] S4 : Rapport et présentation

---

## Bonnes pratiques et règles du projet

- **Jamais de fuite de la vérité terrain** vers le VLM, ni dans l'image ni dans le prompt.
- **Commencer petit** : valider toute la chaîne sur quelques images avec `moondream` avant le benchmark complet.
- **Tout journaliser dans un CSV** (modèle, id, surrogate, score, latence, VRAM). Un résultat non enregistré est un résultat perdu.
- **Fixer les graines** aléatoires, **figer les versions** des modèles et des bibliothèques.
- **Surveiller la VRAM** : décharger un modèle (`ollama stop`) avant d'en charger un plus gros.
- **Éviter le biais de prompt** : une consigne suggestive pousse le modèle à toujours approuver. Tester des formulations neutres.
- **Corrélation ≠ causalité** : un juge corrélé au surrogate n'est pas forcément cliniquement fiable.
- **Commits réguliers** et messages explicites.

---

## Confidentialité et éthique

- **Confidentialité** : les données médicales ne quittent pas la machine. C'est la raison d'être des VLM *locaux* : aucune API externe, aucun envoi d'image.
- **Aucune donnée patient dans ce dépôt** : seuls le code, les configurations et les résultats agrégés sont versionnés.
- **Éthique** : le juge VLM est un **outil d'aide à la recherche**, pas un dispositif de décision clinique. Aucun score ne doit être présenté comme une vérité médicale. Les biais (artéfacts, population, formulation du prompt) sont discutés dans le rapport.

---

## Références

- Castro *et al.*, « Causality matters in medical imaging », *Nature Communications*, 2020.
- Bourbonne, Jaouen *et al.*, « Dosimetric Validation of a GAN-Based Pseudo-CT Generation for MRI-Only Stereotactic Brain Radiotherapy », *Cancers*, 2021.
- Vishnevskiy *et al.*, « Isotropic Total Variation Regularization of Displacements in Parametric Image Registration », *IEEE TMI*, 2017.
- Jaouen *et al.*, « Regularized directional representations for medical image registration », 2021.
- Chen *et al.*, « Large-Scale Label Quality Assessment for Medical Segmentation via a Vision-Language Judge and Synthetic Data » (SegAE), *ISBI 2026*, [arXiv:2601.14406](https://arxiv.org/abs/2601.14406).
- Hering *et al.*, « Learn2Reg », *IEEE TMI*, 2023.
- Huijben *et al.*, « SynthRAD2023 challenge report », *Medical Image Analysis*, 2024.
- Menze *et al.*, « The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS) », *IEEE TMI*, 2015.

**Outils :** [MONAI](https://monai.io) · [SimpleITK](https://simpleitk.org) · [Ollama](https://ollama.com) · [itk-elastix](https://github.com/InsightSoftwareConsortium/ITKElastix)

---

## Équipe et encadrement

- **Encadrant :** Vincent Jaouen, IMT Atlantique, Département Data Science / LaTIM, UMR 1101
- **Équipe (Sujet A) :** Pierre GUTTIEREZ, Rime AOUINA, Florian LAFFONT, Elias HASNAOU, Nourane KHALFALLI, Gerhaldine SUAREZ et  Moemen ZGUEM 

Projet réalisé dans le cadre de l'UE **Projet Commande Entreprise**, IMT Atlantique, 2026.

---

## Glossaire

| Terme | Définition |
|---|---|
| **VLM** | *Vision-Language Model* : modèle qui regarde une image et répond en texte |
| **sCT / pseudo-CT** | CT synthétique généré depuis une autre modalité (IRM, CBCT) |
| **Recalage** | estimation d'un champ de déplacement qui aligne deux images |
| **Oracle** | méthode de référence (ici conventionnelle) qui produit des champs de bonne qualité |
| **Surrogate** | indicateur indirect de qualité utilisé faute de vérité terrain |
| **TRE** | *Target Registration Error* : écart entre points de repère après recalage (mm) |
| **Jacobien** | déterminant du gradient du champ ; un pliage apparaît s'il est négatif |
| **Gamma analysis** | test de concordance dosimétrique (par ex. 2 %/2 mm) |
| **DVH** | histogramme dose–volume |
| **MAE / PSNR / SSIM** | métriques de fidélité d'image |
| **Spearman / AUROC** | métriques pour évaluer la qualité d'un juge |
| **Structured Outputs** | contrainte de sortie JSON via un schéma (paramètre `format` d'Ollama) |
| **VRAM** | mémoire de la carte graphique |
