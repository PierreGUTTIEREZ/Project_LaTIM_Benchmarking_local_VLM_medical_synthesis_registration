# Rapport du volet Synthèse

Ce dossier contient le rapport scientifique (`rapport.pdf`), sa source LaTeX et le diaporama de restitution
(`restitution.pptx`). Les figures, les tableaux et les chiffres du texte sont régénérés à partir de `../results/`.

| Fichier | Rôle |
|---|---|
| `rapport.tex` | texte du rapport (français, XeLaTeX) |
| `references.bib` | bibliographie |
| `make_figures.py` | lit `../results/` et écrit `figures/`, `tables/` et `chiffres.tex` |
| `make_example_figure.py` | figure d'exemples d'images (à lancer **sur la station**) |
| `figures/`, `tables/`, `chiffres.tex` | fichiers générés : ne pas les modifier à la main |
| `restitution.pptx` | diaporama de restitution (17 diapositives, notes de l'orateur incluses) |
| `slides/` | `make_slides.js` (génère le diaporama), `data.json` (chiffres, écrit par `make_figures.py`), `img/` |

## Régénérer le rapport

Prérequis : une distribution LaTeX avec XeLaTeX, `polyglossia`, `fontspec`, `unicode-math`, `tcolorbox`, `natbib`,
`cleveref`, `adjustbox` et `latexmk` (TeX Live complet ou MiKTeX).

```bash
cd Synthesis/report
uv run python make_figures.py        # environ 1 min
latexmk -xelatex rapport.tex         # produit rapport.pdf
```

`make_figures.py` lit :

- `../results/judgements.csv` ;
- `../results/analysis/{val_gray,val_rgb3,test_gray}/scores_and_metrics.csv` et `leaderboard.csv`, produits par
  `pipeline/analyze.py`.

## Figure d'exemples (facultatif)

Les PNG du banc d'essai et les volumes SynthRAD ne quittent pas la station. Pour ajouter au rapport une figure qui
montre le vrai CT et cinq sCT d'une même coupe, lancer sur la station :

```bash
cd ~/<dépôt>/Synthesis/report
uv run python make_example_figure.py          # test, rendu gray, 2 patients
```

Rapatrier ensuite `figures/exemples.pdf` et `figures/exemples.png` sur le PC (`scp`), puis recompiler :
`rapport.tex` inclut la figure automatiquement si elle existe. Les images viennent de SynthRAD2023 (licence de
recherche) : vérifier avec l'encadrant avant de diffuser le rapport hors de l'équipe.

## Après avoir complété le benchmark de test

Le test n'a été lancé que pour 4 modèles. Une fois les autres modèles passés (`bash run_all.sh test gray` sur la
station, puis rapatriement de `results/`) :

1. relancer `uv run python make_figures.py`, puis `latexmk -xelatex rapport.tex` ;
2. relire les sections 4.6 (test), 5 (discussion) et l'encadré « Limite » : le texte de discussion est écrit à la main
   et ne se met pas à jour tout seul.

## Régénérer le diaporama

Après `make_figures.py` (qui écrit aussi `slides/data.json`) :

```bash
cd slides
for f in banc_essai surrogates distributions_val_gray scores_generateurs; do
  pdftoppm -png -r 300 -singlefile ../figures/$f.pdf img/$f
done
node make_slides.js        # écrit ../restitution.pptx
```

Le script demande `pptxgenjs`, `jszip`, `react`, `react-dom`, `react-icons` et `sharp` (npm), ainsi que la variable
`PPTX_SKILL_DIR` qui pointe vers le dossier contenant `scripts/apply_theme.js`. Le plus simple est de demander à Claude
de le relancer. Le diaporama peut aussi être modifié directement dans PowerPoint.
