"""Figures, tables et chiffres du rapport, régénérés à partir des résultats.

    cd Synthesis/report
    uv run python make_figures.py          # puis : latexmk -xelatex rapport.tex

Lit ../results/analysis/<split>_<style>/ (produits par pipeline/analyze.py) et écrit :

    figures/*.pdf     figures vectorielles incluses dans rapport.tex
    figures/*.png     les mêmes en image, pour la présentation
    tables/*.tex      tableaux LaTeX (booktabs)
    chiffres.tex      macros \\newcommand des chiffres cités dans le texte

Si le benchmark de test est complété plus tard, relancer ce script puis recompiler : tableaux,
figures et chiffres se mettent à jour. Relire alors le texte de la discussion.

Toutes les statistiques suivent pipeline/analyze.py : corrélations orientées (+1 = le VLM note
mieux les meilleurs sCT), intervalles de confiance à 95 % par bootstrap sur les patients.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "analysis"
FIGURES, TABLES = HERE / "figures", HERE / "tables"
SEED, N_BOOT = 42, 1000
LOWER_IS_BETTER = {"mae_hu", "mae_hu_slice"} | {
    f"{m}_{mod}" for m in ("dose_diff", "ptv_d98", "ptv_d2", "body_dmean") for mod in ("photons", "protons")}
ACCEPTABLE_HU = 100
# Le rapport porte sur une seule consigne : si d'autres ont été testées sur val, on les écarte ici.
PROMPT = "realism_v1"

# Palette catégorielle validée (skill dataviz), dans l'ordre fixe.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8d8c86", "#e4e3df"
STYLE_COLOR = {"gray": BLUE, "rgb3": ORANGE}
STYLE_LABEL = {"gray": "rendu gray", "rgb3": "rendu rgb3"}

GENERATORS = [f"unet_baseline/epoch_{e:03d}" for e in (1, 2, 5, 10, 20, 50, 100)] + [
    "fedsynth/li", "fedsynth/fu", "fedsynth/spadea"]
GEN_LABEL = {g: f"U-Net ép. {int(g[-3:])}" for g in GENERATORS[:7]} | {
    "fedsynth/li": "Li", "fedsynth/fu": "Fu", "fedsynth/spadea": "Spadea"}

MODEL_INFO = {  # nom affiché, paramètres (milliards), famille
    "moondream:latest": ("Moondream", 1.8, "généraliste"),
    "llava:7b": ("LLaVA 7B", 7, "généraliste"),
    "llava:13b": ("LLaVA 13B", 13, "généraliste"),
    "minicpm-v:8b": ("MiniCPM-V 8B", 8, "généraliste"),
    "qwen3-vl:8b": ("Qwen3-VL 8B", 8, "généraliste"),
    "qwen3-vl:30b": ("Qwen3-VL 30B", 30, "généraliste"),
    "qwen3.8:27b": ("Qwen3.8 27B", 27, "généraliste"),
    "gemma4:latest": ("Gemma 4", None, "généraliste"),  # taille non relevée
    "medgemma:latest": ("MedGemma 4B", 4, "médical"),
    "medgemma:27b": ("MedGemma 27B", 27, "médical"),
}


def name(model):
    return MODEL_INFO.get(model, (model,))[0]


# --- Données ----------------------------------------------------------------------------

def load(split, style):
    path = RESULTS / f"{split}_{style}" / "scores_and_metrics.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path, dtype={"error": str})
    data = data[(data["error"].isna() | (data["error"] == "")) & (data["prompt_name"] == PROMPT)]
    return data


def leaderboard(split, style):
    path = RESULTS / f"{split}_{style}" / "leaderboard.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return table[table["prompt_name"] == PROMPT]


# --- Statistiques (mêmes définitions que pipeline/analyze.py) ---------------------------------

def corr(scores, values, metric):
    scores, values = np.asarray(scores, float), np.asarray(values, float)
    keep = np.isfinite(scores) & np.isfinite(values)
    scores, values = scores[keep], values[keep]
    if len(scores) < 3:
        return np.nan
    if np.ptp(scores) == 0 or np.ptp(values) == 0:
        return 0.0
    rho = spearmanr(scores, values)[0]
    return -rho if metric in LOWER_IS_BETTER else rho


def auroc(scores, positive):
    scores, positive = np.asarray(scores, float), np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def boot_patients(data, statistic, rng, n=N_BOOT):
    """Distribution bootstrap d'une statistique, en tirant les patients avec remise."""
    rows_of = data.groupby("patient_id").indices
    patients = list(rows_of)
    out = []
    for _ in range(n):
        drawn = rng.choice(len(patients), size=len(patients), replace=True)
        rows = np.concatenate([rows_of[patients[i]] for i in drawn])
        out.append(statistic(data.iloc[rows]))
    return np.array(out)


def ci(samples):
    samples = samples[np.isfinite(samples)]
    return (np.percentile(samples, 2.5), np.percentile(samples, 97.5)) if len(samples) else (np.nan, np.nan)


def kept_mean(scores, values, fraction):
    """Moyenne de la métrique sur la part `fraction` la mieux notée (ex aequo partagés au prorata)."""
    n_keep = max(1, int(round(fraction * len(values))))
    cutoff = np.sort(scores)[::-1][n_keep - 1]
    above, tied = scores > cutoff, scores == cutoff
    weights = above.astype(float)
    weights[tied] = (n_keep - above.sum()) / tied.sum()
    return (weights * values).sum() / n_keep


# --- Mise en forme -----------------------------------------------------------------------------

def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlecolor": INK, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
    })


def french_ticks(figure):
    """Virgule décimale et vrai signe moins sur les axes numériques."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    formatter = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    for ax in figure.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(formatter)


def dec(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "\u2212")


def save(figure, stem):
    french_ticks(figure)
    figure.savefig(FIGURES / f"{stem}.pdf")
    figure.savefig(FIGURES / f"{stem}.png", dpi=200)
    plt.close(figure)


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "$-$")


def write_table(stem, header, rows, align, caption_note=None):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(r"\midrule" if row == "MIDRULE" else " & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Figures et tables -------------------------------------------------------------------------

def testbench_figure(volumes):
    """(a) MAE et (b) taux gamma protons de chaque générateur, sur les 27 patients de test."""
    figure, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, metric, label, color in [(axes[0], "mae_hu", "MAE du volume (HU)", BLUE),
                                     (axes[1], "gamma_protons", "Taux gamma 2 %/2 mm, protons (%)", AQUA)]:
        values = [volumes.loc[volumes.generator == g, metric].dropna().values for g in GENERATORS]
        parts = ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=True,
                           medianprops={"color": INK, "linewidth": 1},
                           flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MUTED,
                                       "markeredgecolor": "none"},
                           whiskerprops={"color": INK2, "linewidth": 0.8}, capprops={"color": INK2, "linewidth": 0.8})
        for i, box in enumerate(parts["boxes"]):
            box.set_facecolor(color if i < 7 else ORANGE)
            box.set_alpha(0.55)
            box.set_edgecolor(INK2)
            box.set_linewidth(0.8)
        ax.set_xticks(range(1, 11), [GEN_LABEL[g].replace("U-Net ép. ", "ép. ") for g in GENERATORS],
                      rotation=55, ha="right")
        ax.set_ylabel(label)
        ax.axvline(7.5, color=MUTED, linewidth=0.6, linestyle=":")
    axes[0].text(4, axes[0].get_ylim()[1] * 0.97, "U-Net (checkpoints)", ha="center", va="top", color=INK2, fontsize=7)
    axes[0].text(9, axes[0].get_ylim()[1] * 0.97, "pré-entraînés", ha="center", va="top", color=INK2, fontsize=7)
    figure.tight_layout()
    save(figure, "banc_essai")


def generator_table(volumes, stem):
    rows = []
    for g in GENERATORS:
        d = volumes[volumes.generator == g]
        rows.append([GEN_LABEL[g], len(d),
                     f"{d.mae_hu.mean():.0f} $\\pm$ {d.mae_hu.std():.0f}", fmt(d.psnr_db.mean(), 1),
                     fmt(d.ssim.mean(), 3), fmt(d.gamma_photons.mean(), 1), fmt(d.gamma_protons.mean(), 1),
                     fmt(d.dose_diff_protons.mean(), 2), fmt(d.ptv_d98_protons.mean(), 2)])
        if g == GENERATORS[6]:
            rows.append("MIDRULE")
    write_table(stem, ["Générateur", "$n$", "MAE (HU)", "PSNR (dB)", "SSIM", r"$\gamma_\text{ph}$ (\%)",
                       r"$\gamma_\text{pr}$ (\%)", r"$\Delta D_\text{pr}$ (\%)", r"$\Delta D98_\text{pr}$ (\%)"],
                rows, "lrrrrrrrr")


def surrogate_figure(volumes):
    """Corrélations de Spearman entre métriques de référence, au niveau du sCT (val + test)."""
    metrics = [("mae_hu", "MAE"), ("psnr_db", "PSNR"), ("ssim", "SSIM"),
               ("gamma_photons", "γ photons"), ("dose_diff_photons", "ΔD photons"), ("ptv_d98_photons", "ΔD98 photons"),
               ("gamma_protons", "γ protons"), ("dose_diff_protons", "ΔD protons"), ("ptv_d98_protons", "ΔD98 protons"),
               ("ptv_d2_protons", "ΔD2 protons")]
    keys = [m for m, _ in metrics if m in volumes]
    labels = [lab for m, lab in metrics if m in volumes]
    n = len(keys)
    matrix = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            rho = spearmanr(volumes[a], volumes[b], nan_policy="omit")[0]
            # Orientation : +1 = les deux métriques désignent les mêmes sCT comme meilleurs.
            flip = (a in LOWER_IS_BETTER) != (b in LOWER_IS_BETTER)
            matrix[i, j] = -rho if flip else rho
    figure, ax = plt.subplots(figsize=(4.6, 3.9))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f3f2ee", BLUE])
    image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.3,
                    color=INK if abs(matrix[i, j]) < 0.75 else "white")
    ax.set_xticks(range(n), labels, rotation=55, ha="right")
    ax.set_yticks(range(n), labels)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = figure.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("ρ de Spearman orienté")
    bar.outline.set_visible(False)
    figure.tight_layout()
    save(figure, "surrogates")
    return dict(zip([(a, b) for a in keys for b in keys], matrix.flatten()))


def distribution_figure(data, stem, order):
    """Histogramme des notes de chaque VLM (validation, rendu gray)."""
    columns = 5
    lines = int(np.ceil(len(order) / columns))
    figure, axes = plt.subplots(lines, columns, figsize=(6.3, 1.45 * lines + 0.2), sharex=True, squeeze=False)
    bins = np.arange(-2.5, 103, 5)
    for ax, model in zip(axes.flat, order):
        d = data[data.model == model]
        ax.hist(d.score, bins=bins, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(name(model), fontsize=7.5)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Note (0–100)")
    figure.tight_layout(h_pad=0.6)
    save(figure, stem)


def judge_rows(data, rng):
    """Statistiques par VLM : ρ et IC, intra, AUROC, filtrage, ρ niveau générateur."""
    out = {}
    for model, d in data.groupby("model"):
        d = d.reset_index(drop=True)
        samples = boot_patients(d, lambda x: corr(x.score, x.mae_hu_slice, "mae_hu_slice"), rng)
        gen = d.groupby("generator").agg(score=("score", "mean"), mae=("mae_hu", "mean"))
        out[model] = {
            "n": len(d),
            "rho": corr(d.score, d.mae_hu_slice, "mae_hu_slice"),
            "ci": ci(samples),
            "auroc": auroc(d.score, d.mae_hu_slice < ACCEPTABLE_HU),
            "std": d.score.std(ddof=0),
            "unique": d.score.nunique(),
            "rho_gen": corr(gen.score, gen.mae, "mae_hu"),
            "gain": None,
            "latency": d.latency_s.median(),
            "vram": d.vram_gb.max(),
        }
        values, scores = d.mae_hu_slice.to_numpy(float), d.score.to_numpy(float)
        best = np.sort(values)[: len(values) // 2].mean()
        out[model]["gain"] = (values.mean() - kept_mean(scores, values, 0.5)) / (values.mean() - best)
    return out


def leaderboard_table(stats, lb, stem, models):
    rows = []
    for model in models:
        s = stats.get(model)
        line = lb[lb.model == model].iloc[0] if lb is not None and (lb.model == model).any() else None
        if s is None or line is None:
            continue
        cov = line["coverage"]
        rows.append([name(model), f"{100 * cov:.0f}", fmt(s["std"], 1),
                     f"{fmt(s['rho'])} [{fmt(s['ci'][0])} ; {fmt(s['ci'][1])}]",
                     fmt(line["spearman_intra"]), fmt(s["rho_gen"]), fmt(s["auroc"]), fmt(s["gain"]),
                     fmt(s["latency"], 2), fmt(s["vram"], 1)])
    write_table(stem, ["VLM", r"Couv. (\%)", r"$\sigma$ note", r"$\rho$ [IC 95\,\%]", r"$\rho_\text{intra}$",
                       r"$\rho_\text{gén}$", "AUROC", "Gain filtr.", "Lat. (s)", "VRAM (Go)"],
                rows, "lrrcrrrrrr")


def forest_figure(stats_by_style, order):
    figure, ax = plt.subplots(figsize=(6.3, 3.3))
    y = np.arange(len(order))[::-1]
    for offset, style in [(0.17, "gray"), (-0.17, "rgb3")]:
        stats = stats_by_style.get(style, {})
        for yi, model in zip(y, order):
            s = stats.get(model)
            if s is None:
                continue
            low, high = s["ci"]
            ax.plot([low, high], [yi + offset] * 2, color=STYLE_COLOR[style], linewidth=1.6, solid_capstyle="round")
            ax.plot(s["rho"], yi + offset, "o", color=STYLE_COLOR[style], markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8, label=STYLE_LABEL[style] if yi == y[0] else None)
    ax.axvline(0, color=INK2, linewidth=0.8)
    def label(model):
        low_coverage = any(stats_by_style[s].get(model, {"n": 810})["n"] < 0.9 * 810 for s in stats_by_style)
        return name(model) + (" *" if low_coverage else "")
    ax.set_yticks(y, [label(m) for m in order])
    ax.set_xlabel("ρ de Spearman orienté entre note et MAE de la coupe (IC 95 %, bootstrap patients)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    figure.tight_layout()
    save(figure, "leaderboard_val")


def style_delta_table(val, rng, models):
    gray, rgb3 = val["gray"], val["rgb3"]
    rows, raw = [], []
    for model in models:
        a = gray[gray.model == model][["item_id", "patient_id", "score", "mae_hu_slice"]]
        b = rgb3[rgb3.model == model][["item_id", "score"]].copy()
        b["item_id"] = b.item_id.str.replace("_rgb3", "_gray", regex=False)
        both = a.merge(b, on="item_id", suffixes=("_g", "_r")).reset_index(drop=True)
        if len(both) < 30:
            continue
        stat = lambda x: corr(x.score_r, x.mae_hu_slice, "mae_hu_slice") - corr(x.score_g, x.mae_hu_slice, "mae_hu_slice")
        delta = stat(both)
        low, high = ci(boot_patients(both, stat, rng))
        verdict = "rgb3" if low > 0 else ("gray" if high < 0 else "--")
        raw.append({"model": model, "name": name(model), "n": len(both), "delta": delta, "low": low, "high": high})
        rows.append([name(model), len(both), fmt(corr(both.score_g, both.mae_hu_slice, "mae_hu_slice")),
                     fmt(corr(both.score_r, both.mae_hu_slice, "mae_hu_slice")),
                     f"{fmt(delta)} [{fmt(low)} ; {fmt(high)}]", verdict])
    write_table("delta_style", ["VLM", "$n$ appariées", r"$\rho$ gray", r"$\rho$ rgb3",
                                r"$\Delta\rho$ (rgb3 $-$ gray) [IC 95\,\%]", "Meilleur rendu"], rows, "lrrrcc")
    return raw


def pairwise_table(data, reference, rng, models):
    ref = data[data.model == reference][["item_id", "patient_id", "score", "mae_hu_slice"]]
    rows = []
    for model in models:
        if model == reference:
            continue
        other = data[data.model == model][["item_id", "score"]]
        both = ref.merge(other, on="item_id", suffixes=("_a", "_b")).reset_index(drop=True)
        if len(both) < 30:
            continue
        stat = lambda x: corr(x.score_a, x.mae_hu_slice, "mae_hu_slice") - corr(x.score_b, x.mae_hu_slice, "mae_hu_slice")
        low, high = ci(boot_patients(both, stat, rng))
        rows.append([name(model), len(both), f"{fmt(stat(both))} [{fmt(low)} ; {fmt(high)}]",
                     "oui" if low > 0 else "non"])
    write_table("pairwise", ["Comparé à", "$n$ appariées", rf"$\rho_\text{{{name(reference)}}} - \rho_\text{{VLM}}$ [IC 95\,\%]",
                             "Différence significative"], rows, "lrcc")


def generator_score_figure(panels):
    """Note moyenne par générateur (IC 95 % bootstrap patients) en fonction de la MAE moyenne."""
    figure, axes = plt.subplots(1, len(panels), figsize=(6.3, 2.35), squeeze=False)
    rng = np.random.default_rng(SEED)
    for ax, (data, model, style) in zip(axes[0], panels):
        d = data[data.model == model]
        for g in GENERATORS:
            dg = d[d.generator == g].reset_index(drop=True)
            if dg.empty:
                continue
            mae = dg.drop_duplicates("patient_id").mae_hu.mean()
            low, high = ci(boot_patients(dg, lambda x: x.score.mean(), rng, n=400))
            color = BLUE if g.startswith("unet") else ORANGE
            ax.plot([mae, mae], [low, high], color=color, linewidth=1.2)
            ax.plot(mae, dg.score.mean(), "o", color=color, markersize=4.5, markeredgecolor="white", markeredgewidth=0.7)
            if g in ("unet_baseline/epoch_001", "unet_baseline/epoch_100", "fedsynth/li"):
                ax.annotate(GEN_LABEL[g].replace("U-Net ", ""), (mae, dg.score.mean()), textcoords="offset points",
                            xytext=(4, 3), fontsize=6.5, color=INK2)
        ax.set_title(f"{name(model)} · {style}")
        ax.set_xlabel("MAE moyenne du générateur (HU)")
    axes[0][0].set_ylabel("Note moyenne du VLM")
    axes[0][0].plot([], [], "o", color=BLUE, label="U-Net (checkpoints)")
    axes[0][0].plot([], [], "o", color=ORANGE, label="Li, Fu, Spadea")
    axes[0][0].legend(loc="lower left", fontsize=6.5)
    figure.tight_layout()
    save(figure, "scores_generateurs")


def filtering_figure(panels):
    fractions = np.linspace(0.1, 1.0, 19)
    figure, axes = plt.subplots(1, len(panels), figsize=(6.3, 2.5), squeeze=False, sharey=True)
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    for ax, (title, data, models) in zip(axes[0], panels):
        reference = data[data.model == models[0]]
        values = reference.mae_hu_slice.to_numpy(float)
        ax.plot(fractions * 100, [np.sort(values)[: max(1, int(round(f * len(values))))].mean() for f in fractions],
                color=INK2, linestyle="--", linewidth=1.2, label="oracle (vraie MAE)")
        ax.axhline(values.mean(), color=MUTED, linestyle=":", linewidth=1.2, label="hasard")
        for color, model in zip(colors, models):
            d = data[data.model == model]
            v, s = d.mae_hu_slice.to_numpy(float), d.score.to_numpy(float)
            ax.plot(fractions * 100, [kept_mean(s, v, f) for f in fractions], color=color, linewidth=2,
                    label=name(model))
        ax.set_title(title)
        ax.set_xlabel("Part des coupes gardées (%)")
        ax.legend(loc="lower right", fontsize=6.5)
    axes[0][0].set_ylabel("MAE moyenne des coupes gardées (HU)")
    figure.tight_layout()
    save(figure, "filtrage")


def roc_figure(data, models):
    figure, ax = plt.subplots(figsize=(3.1, 2.9))
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    for color, model in zip(colors, models):
        d = data[data.model == model]
        positive = (d.mae_hu_slice < ACCEPTABLE_HU).to_numpy()
        scores = d.score.to_numpy(float)
        thresholds = np.unique(scores)[::-1]
        tpr = [0] + [((scores >= t) & positive).sum() / positive.sum() for t in thresholds]
        fpr = [0] + [((scores >= t) & ~positive).sum() / (~positive).sum() for t in thresholds]
        ax.plot(fpr, tpr, color=color, linewidth=1.8, drawstyle="default",
                label=f"{name(model)} ({dec(auroc(scores, positive))})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6.3, title="AUROC", title_fontsize=6.5)
    figure.tight_layout()
    save(figure, "roc")


def cost_figure(stats_by_style):
    figure, ax = plt.subplots(figsize=(6.3, 2.8))
    best = {}
    for style, stats in stats_by_style.items():
        for model, s in stats.items():
            if s["n"] < 100:
                continue
            ax.plot(s["latency"], s["rho"], "o", color=STYLE_COLOR[style], markersize=3 + s["vram"] / 3.5,
                    alpha=0.8, markeredgecolor="white", markeredgewidth=0.8)
            if model not in best or s["rho"] > best[model][1]:
                best[model] = (s["latency"], s["rho"])
    offsets = {"qwen3.8:27b": (6, 6), "qwen3-vl:8b": (-12, -12), "medgemma:27b": (-62, 5),
               "gemma4:latest": (5, -10), "llava:13b": (6, 4)}
    for model, (latency, rho) in best.items():
        if rho > 0.1:
            ax.annotate(name(model), (latency, rho), textcoords="offset points", xytext=offsets.get(model, (7, 4)),
                        fontsize=6.5, color=INK2)
    for style in stats_by_style:
        ax.plot([], [], "o", color=STYLE_COLOR[style], label=STYLE_LABEL[style])
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xscale("log")
    ticks = [0.1, 0.2, 0.5, 1, 2, 5]
    ax.set_xticks(ticks, [dec(v, 1) if v < 1 else str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Latence médiane par image (s, échelle log) — taille du point ∝ VRAM")
    ax.set_ylabel("ρ orienté (MAE coupe)")
    ax.legend(loc="upper left")
    figure.tight_layout()
    save(figure, "cout")


def val_test_figure(val_stats, test_stats):
    common = [m for m in val_stats if m in test_stats and test_stats[m]["n"] >= 100]
    common.sort(key=lambda m: val_stats[m]["rho"])
    figure, ax = plt.subplots(figsize=(3.1, 1.9))
    for i, model in enumerate(common):
        for stats, color, dy in [(val_stats, BLUE, 0.12), (test_stats, ORANGE, -0.12)]:
            low, high = stats[model]["ci"]
            ax.plot([low, high], [i + dy] * 2, color=color, linewidth=1.4)
            ax.plot(stats[model]["rho"], i + dy, "o", color=color, markersize=4, markeredgecolor="white")
    ax.plot([], [], "o", color=BLUE, label="validation")
    ax.plot([], [], "o", color=ORANGE, label="test")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(common)), [name(m) for m in common])
    ax.set_xlabel("ρ orienté (rendu gray)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.5)
    figure.tight_layout()
    save(figure, "val_test")
    return common


NO_ARTIFACT = (r"\bno (?:visible |obvious |apparent |significant |noticeable |major )?(?:artifacts?|artefacts?)\b"
               r"|without (?:any )?(?:visible |obvious )?artifacts?|free of artifacts")


def keyword_figure(data, models_absence, models_checker):
    """Ce que disent les justifications : (a) « pas d'artefact » affirmé, (b) damier signalé."""
    figure, axes = plt.subplots(1, 2, figsize=(6.3, 2.75), sharey=False)
    x = np.arange(10)
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    for color, model in zip(colors, models_absence):
        d = data[data.model == model]
        share = [100 * d[d.generator == g].reason.str.contains(NO_ARTIFACT, case=False, regex=True).mean()
                 for g in GENERATORS]
        axes[0].plot(x, share, "o-", color=color, linewidth=1.6, markersize=3.5, label=name(model))
    axes[0].set_ylabel("Justifications (%)")
    axes[0].set_title("(a) affirment l'absence d'artefact")
    axes[0].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    for color, model in zip([BLUE, VIOLET], models_checker):
        d = data[data.model == model]
        said = d.reason.str.contains("checkerboard", case=False) & ~d.reason.str.contains(r"no[^.]{0,60}checkerboard", case=False)
        share = [100 * said[d.generator == g].mean() for g in GENERATORS]
        axes[1].plot(x, share, "o-", color=color, linewidth=1.6, markersize=3.5, label=name(model))
    axes[1].set_title("(b) signalent un motif en damier")
    axes[1].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    for ax in axes:
        ax.set_xticks(x, [GEN_LABEL[g].replace("U-Net ép. ", "ép. ") for g in GENERATORS], rotation=55, ha="right")
        ax.axvline(6.5, color=MUTED, linewidth=0.6, linestyle=":")
    figure.tight_layout()
    save(figure, "justifications")


def error_table(judgements):
    """Échecs techniques par VLM : images notées du premier coup, après un nouvel essai, ou jamais."""
    j = judgements.copy()
    j["error"] = j["error"].fillna("")
    j.loc[(j["error"] == "") & (j["reason"].fillna("").str.strip() == ""), "error"] = "justification vide"
    j["attempt"] = pd.to_numeric(j["attempt"], errors="coerce").fillna(1) if "attempt" in j else 1
    kinds = {"coupée (plafond)": "réponse coupée", "boucle (Ollama)": "token repeat limit",
             "justif. vide": "justification vide"}
    rows = []
    for model, d in j.groupby("model"):
        per_item = d.groupby(["prompt_name", "item_id"])
        ok_first = per_item.apply(lambda g: bool(((g.error == "") & (g.attempt <= 1)).any() and (g.error != "").sum() == 0))
        ok_any = per_item.apply(lambda g: bool((g.error == "").any()))
        if ok_first.all():
            continue
        last_error = d[d.error != ""].groupby(["prompt_name", "item_id"]).error.last()
        failed = last_error[~ok_any.reindex(last_error.index).astype(bool)]
        counts = [int(failed.str.contains(pattern).sum()) for pattern in kinds.values()]
        n = len(ok_any)
        rows.append([name(model), n, int(ok_first.sum()), int((ok_any & ~ok_first).sum()), len(failed), *counts,
                     fmt(100 * len(failed) / n, 1)])
    write_table("erreurs", ["VLM", "Images", "1\\textsuperscript{er} essai", "Rattrapées", "En échec", *kinds.keys(),
                            r"Échec (\%)"], rows, "lrrrrrrrr")


def position_table(data, models):
    rows = []
    for model in models:
        d = data[data.model == model]
        rows.append([name(model)] + [fmt(corr(d[d.slice_position == p].score, d[d.slice_position == p].mae_hu_slice,
                                              "mae_hu_slice")) for p in (0.35, 0.5, 0.65)])
    write_table("position", ["VLM", "coupe à 35 \\%", "coupe à 50 \\%", "coupe à 65 \\%"], rows, "lrrr")


def dose_corr_table(data, models):
    metrics = [("mae_hu", "MAE vol."), ("ssim", "SSIM vol."), ("gamma_photons", r"$\gamma_\text{ph}$"),
               ("gamma_protons", r"$\gamma_\text{pr}$"), ("dose_diff_protons", r"$\Delta D_\text{pr}$"),
               ("ptv_d98_protons", r"$\Delta D98_\text{pr}$")]
    rows = []
    for model, style, d in models:
        rows.append([f"{name(model)} ({style})" if "," in style else f"{name(model)} (val, {style})"] + [fmt(corr(d[d.model == model].score, d[d.model == model][m], m))
                                                    for m, _ in metrics])
    write_table("dose_corr", ["VLM (rendu)"] + [lab for _, lab in metrics], rows, "lrrrrrr")


def center_table(volumes):
    rows = []
    for g in ["unet_baseline/epoch_020", "unet_baseline/epoch_050", "unet_baseline/epoch_100",
              "fedsynth/li", "fedsynth/fu", "fedsynth/spadea"]:
        d = volumes[volumes.generator == g]
        by = d.groupby("center").mae_hu.mean()
        rows.append([GEN_LABEL[g], fmt(by.get("A"), 0), fmt(by.get("B"), 0), fmt(by.get("C"), 0),
                     fmt(by.get("C") / by.get("A"), 2)])
    write_table("centres", ["Générateur", "Centre A", "Centre B", "Centre C", "Rapport C / A"], rows, "lrrrr")


# --- Programme principal ---------------------------------------------------------------------------

def export_slide_data(stats, test_stats, order, deltas, common, val, test, best_val):
    """Chiffres du diaporama (slides/data.json), calculés exactement comme ceux du rapport."""
    import json

    def judge(s):
        return {"rho": s["rho"], "low": s["ci"][0], "high": s["ci"][1], "n": s["n"], "gen": s["rho_gen"],
                "auroc": s["auroc"], "gain": s["gain"], "latency": s["latency"], "vram": s["vram"]}

    fractions = [round(f, 2) for f in np.linspace(0.1, 1.0, 10)]
    curves = {}
    for split, data, models in [("val", val["gray"], [best_val, "medgemma:latest", "llava:7b"]),
                                ("test", test, ["medgemma:latest", "llava:7b", "gemma4:latest"])]:
        values = data[data.model == models[0]].mae_hu_slice.to_numpy(float)
        curve = {"fractions": fractions, "hasard": [values.mean()] * len(fractions),
                 "oracle": [np.sort(values)[: max(1, int(round(f * len(values))))].mean() for f in fractions]}
        for model in models:
            d = data[data.model == model]
            curve[name(model)] = [kept_mean(d.score.to_numpy(float), d.mae_hu_slice.to_numpy(float), f) for f in fractions]
        curves[split] = curve
    metrics = ["mae_hu", "ssim", "gamma_photons", "gamma_protons", "ptv_d98_protons"]
    dose = {name(m): [corr(val["gray"][val["gray"].model == m].score, val["gray"][val["gray"].model == m][k], k)
                      for k in metrics] for m in [best_val, "medgemma:latest"]}
    out = {
        "order": [name(m) for m in order],
        "val": {style: {name(m): judge(s) for m, s in stats[style].items()} for style in stats},
        "test": {name(m): judge(s) for m, s in test_stats.items() if s["n"] >= 100},
        "common": [name(m) for m in common],
        "deltas": deltas,
        "filtering": curves,
        "dose": {"metrics": metrics, "values": dose},
        "best_val": name(best_val),
    }
    folder = HERE / "slides"
    folder.mkdir(exist_ok=True)
    (folder / "data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    setup_matplotlib()
    rng = np.random.default_rng(SEED)

    val = {style: load("val", style) for style in ("gray", "rgb3")}
    test = load("test", "gray")
    if val["gray"] is None or test is None:
        sys.exit("Résultats manquants : lancer d'abord pipeline/analyze.py sur val_gray et test_gray.")
    judgements = pd.read_csv(HERE.parent / "results" / "judgements.csv", dtype={"error": str})
    judgements = judgements[judgements["prompt_name"] == PROMPT]

    # Banc d'essai (une ligne par sCT)
    volumes_test = test.drop_duplicates(["patient_id", "generator"])
    volumes_val = val["gray"].drop_duplicates(["patient_id", "generator"])
    volumes_all = pd.concat([volumes_val, volumes_test])
    testbench_figure(volumes_test)
    generator_table(volumes_test, "generateurs_test")
    generator_table(volumes_val, "generateurs_val")
    center_table(volumes_all)
    surrogates = surrogate_figure(volumes_all)

    # Juges sur val
    stats = {style: judge_rows(val[style], rng) for style in ("gray", "rgb3") if val[style] is not None}
    best_rho = {m: max(stats[s][m]["rho"] for s in stats if m in stats[s]) for m in stats["gray"]}
    order = sorted(best_rho, key=best_rho.get, reverse=True)
    distribution_figure(val["gray"], "distributions_val_gray", order)
    if "rgb3" in stats:
        distribution_figure(val["rgb3"], "distributions_val_rgb3", order)
    for style in stats:
        leaderboard_table(stats[style], leaderboard("val", style), f"leaderboard_val_{style}",
                          sorted(stats[style], key=lambda m: stats[style][m]["rho"], reverse=True))
    forest_figure(stats, order)
    deltas = style_delta_table(val, rng, order)
    best_val = max(stats["gray"], key=lambda m: stats["gray"][m]["rho"])
    pairwise_table(val["gray"], best_val, rng, order)

    # Test
    test_stats = judge_rows(test, rng)
    test_models = sorted([m for m in test_stats if test_stats[m]["n"] >= 100],
                         key=lambda m: test_stats[m]["rho"], reverse=True)
    leaderboard_table(test_stats, leaderboard("test", "gray"), "leaderboard_test", test_models)
    common = val_test_figure(stats["gray"], test_stats)

    # Figures d'analyse
    top_rgb3 = max(stats["rgb3"], key=lambda m: stats["rgb3"][m]["rho"] if m != best_val else -9) if "rgb3" in stats else None
    generator_score_figure([(val["gray"], best_val, "gray"), (val["gray"], "medgemma:latest", "gray"),
                            (val["rgb3"], top_rgb3, "rgb3")])
    filtering_figure([("Validation · gray", val["gray"], [best_val, "medgemma:latest", "llava:7b"]),
                      ("Test · gray", test, ["medgemma:latest", "llava:7b", "gemma4:latest"])])
    roc_figure(val["gray"], [best_val, "medgemma:latest", "llava:7b", "gemma4:latest"])
    cost_figure(stats)
    keyword_figure(val["gray"], [best_val, "medgemma:latest", "llava:7b", "gemma4:latest"], [best_val, "qwen3.8:27b"])
    error_table(judgements)
    position_table(val["gray"], [best_val, "medgemma:latest", "llava:7b", "gemma4:latest"])
    dose_corr_table(None, [(best_val, "gray", val["gray"]), ("medgemma:latest", "gray", val["gray"]),
                           (top_rgb3, "rgb3", val["rgb3"]), ("medgemma:latest", "test, gray", test)])

    # Chiffres cités dans le texte
    def rho_ci(s):
        return f"{s['rho']:.2f} [{s['ci'][0]:.2f}~; {s['ci'][1]:.2f}]".replace(".", ",").replace("-", "$-$")

    g = stats["gray"]
    t = test_stats
    ep1 = volumes_test[volumes_test.generator == GENERATORS[0]].mae_hu.mean()
    ep100 = volumes_test[volumes_test.generator == GENERATORS[6]].mae_hu.mean()
    fed = volumes_test[volumes_test.generator.str.startswith("fedsynth")].mae_hu.mean()
    macros = {
        "nJudgements": f"{len(judgements):,}".replace(",", "\\,"),
        "nErrors": str((judgements.error.notna() & (judgements.error != "")).sum()),
        "nModelsVal": str(len(g)),
        "nModelsTest": str(len(test_models)),
        "pctAcceptable": f"{100 * (volumes_test.mae_hu < ACCEPTABLE_HU).mean():.0f}",
        "pctAcceptableSlices": f"{100 * (test.drop_duplicates('item_id').mae_hu_slice < ACCEPTABLE_HU).mean():.0f}",
        "maeEpOne": f"{ep1:.0f}", "maeEpHundred": f"{ep100:.0f}", "maeFedsynth": f"{fed:.0f}",
        "bestVal": name(best_val), "bestValRho": rho_ci(g[best_val]),
        "bestValIntra": f"{leaderboard('val', 'gray').set_index('model').loc[best_val, 'spearman_intra']:.2f}".replace(".", ","),
        "bestValGen": f"{g[best_val]['rho_gen']:.2f}".replace(".", ","),
        "bestValAuroc": f"{g[best_val]['auroc']:.2f}".replace(".", ","),
        "bestValGain": f"{g[best_val]['gain']:.2f}".replace(".", ","),
        "medValRho": rho_ci(g["medgemma:latest"]),
        "medValGen": f"{g['medgemma:latest']['rho_gen']:.2f}".replace(".", ","),
        "medTestRho": rho_ci(t["medgemma:latest"]),
        "medTestGen": f"{t['medgemma:latest']['rho_gen']:.2f}".replace(".", ","),
        "medTestAuroc": f"{t['medgemma:latest']['auroc']:.2f}".replace(".", ","),
        "medTestGain": f"{t['medgemma:latest']['gain']:.2f}".replace(".", ","),
        "medRgbRho": rho_ci(stats["rgb3"]["medgemma:latest"]) if "rgb3" in stats else "--",
        "medBigStd": f"{g['medgemma:27b']['std']:.1f}".replace(".", ","),
        "corrMaeSsim": f"{surrogates[('mae_hu', 'ssim')]:.2f}".replace(".", ","),
        "corrMaeGammaPh": f"{surrogates[('mae_hu', 'gamma_photons')]:.2f}".replace(".", ","),
        "corrMaeGammaPr": f"{surrogates[('mae_hu', 'gamma_protons')]:.2f}".replace(".", ","),
        "corrGammaPhPr": f"{surrogates[('gamma_photons', 'gamma_protons')]:.2f}".replace(".", ","),
        "commonValTest": str(len(common)),
    }
    (HERE / "chiffres.tex").write_text(
        "% Fichier généré par make_figures.py : ne pas modifier à la main.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")
    export_slide_data(stats, test_stats, order, deltas, common, val, test, best_val)
    print("Meilleur juge sur val (gray) :", best_val, "| meilleur rgb3 hors celui-ci :", top_rgb3)
    print(f"Figures dans {FIGURES}, tables dans {TABLES}, chiffres dans chiffres.tex")


if __name__ == "__main__":
    main()
