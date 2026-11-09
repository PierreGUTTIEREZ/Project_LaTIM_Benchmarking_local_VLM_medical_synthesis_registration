"""Analyse : les notes des VLM suivent-elles la vraie qualité des sCT ?

Rattache les notes (results/judgements.csv) aux métriques du banc d'essai (testbench.csv)
grâce à item_id, puis compare, pour chaque couple (modèle, prompt), les notes aux métriques.

    uv run python analyze.py --testbench ../sct_generation/work/testbench/val/gray/testbench.csv

Écrit dans results/analysis/<split>_<style>/ :

    leaderboard.csv          une ligne par (modèle, prompt), du meilleur juge au moins bon
    correlations.csv         Spearman entre la note et chacune des métriques (coupe et volume)
    filtering.csv            ce que rapporte un filtrage par le juge, comparé à l'oracle et au hasard
    generators.csv           qualité de chaque générateur de sCT (pour décrire le banc d'essai)
    scores_and_metrics.csv   chaque note avec les métriques de l'image notée
    scatter.png              note du VLM en fonction de la métrique de référence, par modèle

Métriques de dose : si sct_generation/dose.py a tourné sur la même partie du split, son fichier
work/testbench/<split>/dose_metrics.csv est lu automatiquement et ses colonnes (gamma_photons,
dose_diff_protons...) sont ajoutées aux métriques analysées. Pas besoin de refaire le benchmark :
les notes des VLM ne changent pas, il suffit de relancer ce script.

Lecture des corrélations : le signe est orienté pour que +1 veuille toujours dire « le VLM
donne les meilleures notes aux meilleurs sCT », 0 « aucun lien » et -1 « l'inverse »
(pour la MAE, où plus bas = meilleur, on change donc le signe). Un juge qui donne toujours
la même note obtient 0.

Colonnes du leaderboard (la métrique de référence est TARGET_METRIC dans config.py) :

    n_images, coverage       images notées sans erreur, et leur part dans le banc d'essai
    n_errors                 images restées en erreur
    score_mean, score_std    moyenne et dispersion des notes (dispersion nulle = juge inutile)
    spearman_<métrique>      corrélation de rang entre note et métrique, sur toutes les images
    ..._ci_low, ..._ci_high  intervalle de confiance à 95 % (bootstrap sur les patients)
    pearson_<métrique>       corrélation linéaire
    spearman_intra           corrélation à anatomie égale : pour chaque coupe d'un patient, on
                             classe les sCT des différents générateurs, puis on fait la moyenne.
                             C'est la mesure la plus stricte : elle ne récompense pas un juge qui
                             réagit seulement au contenu de la coupe (beaucoup d'os, sinus...).
    auroc                    séparation des sCT acceptables / non acceptables (0,5 = hasard)
    latency_s_mean, vram_gb  coût du juge
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'écran sur la station : on écrit directement des PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

import config

IMAGE_METRICS = ["mae_hu_slice", "psnr_db_slice", "ssim_slice", "mae_hu", "psnr_db", "ssim"]
DOSE_METRICS = ["gamma", "dose_diff", "ptv_d98", "ptv_d2", "body_dmean"]  # voir sct_generation/dose.py
METRICS = list(IMAGE_METRICS)  # complétée par les colonnes de dose si elles existent (add_dose_metrics)
METRIC_LABELS = {
    "mae_hu_slice": "MAE de la coupe (HU)",
    "psnr_db_slice": "PSNR de la coupe (dB)",
    "ssim_slice": "SSIM de la coupe",
    "mae_hu": "MAE du volume (HU)",
    "psnr_db": "PSNR du volume (dB)",
    "ssim": "SSIM du volume",
    **{f"gamma_{m}": f"Taux gamma 2 %/2 mm, {m} (%)" for m in ("photons", "protons")},
    **{f"dose_diff_{m}": f"Écart de dose moyen, {m} (% de la prescription)" for m in ("photons", "protons")},
}
JUDGE_KEY = ["model", "prompt_name"]


# --- Lecture des données ------------------------------------------------------------

def load_judgements(path):
    """Renvoie (notes réussies, notes restées en erreur), sans doublons."""
    if not path.exists():
        raise SystemExit(f"{path} introuvable : lancer d'abord benchmark.py.")
    judgements = pd.read_csv(path, dtype={"error": str})
    if judgements.empty:
        raise SystemExit(f"{path} est vide : lancer d'abord benchmark.py.")

    key = JUDGE_KEY + ["item_id"]
    judgements["error"] = judgements["error"].fillna("")
    ok = judgements[judgements["error"] == ""].drop_duplicates(key, keep="last")

    # Une image en erreur puis réussie au second essai ne compte pas comme une erreur.
    failed = judgements[judgements["error"] != ""].drop_duplicates(key, keep="last")
    failed = failed.merge(ok[key], on=key, how="left", indicator=True)
    failed = failed[failed["_merge"] == "left_only"].drop(columns="_merge")
    return ok, failed


def add_dose_metrics(testbench, testbench_csv):
    """Ajoute au banc d'essai les métriques de dose de dose.py, s'il a tourné. Renvoie leurs noms."""
    dose_csv = testbench_csv.parent.parent / "dose_metrics.csv"
    if not dose_csv.exists():
        return testbench, []
    dose = pd.read_csv(dose_csv).drop_duplicates(["patient_id", "generator", "modality"], keep="last")
    wide = dose.pivot_table(index=["patient_id", "generator"], columns="modality", values=DOSE_METRICS)
    wide.columns = [f"{metric}_{modality}" for metric, modality in wide.columns]
    columns = [f"{metric}_{modality}" for modality in sorted(dose["modality"].unique()) for metric in DOSE_METRICS]
    testbench = testbench.merge(wide.reset_index(), on=["patient_id", "generator"], how="left")
    n_with_dose = testbench[columns[0]].notna().sum()
    print(f"Métriques de dose ajoutées depuis {dose_csv.name} : {n_with_dose} / {len(testbench)} images.")
    return testbench, columns


# --- Statistiques -------------------------------------------------------------------

def oriented(value, metric):
    """Change le signe pour les métriques où plus bas = meilleur (voir LOWER_IS_BETTER)."""
    return -value if metric in config.LOWER_IS_BETTER else value


def correlation(scores, values, metric, method=spearmanr):
    """Corrélation orientée entre les notes et une métrique (Spearman par défaut)."""
    scores = np.asarray(scores, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(values)
    scores, values = scores[keep], values[keep]
    if len(scores) < 3:
        return np.nan
    # Des notes toutes identiques ne permettent pas de classer les sCT : aucune information.
    if np.ptp(scores) == 0 or np.ptp(values) == 0:
        return 0.0
    return oriented(float(method(scores, values)[0]), metric)


def bootstrap_interval(data, metric, rng):
    """Intervalle de confiance à 95 % de la corrélation de Spearman, par bootstrap sur les patients.

    On tire des patients et non des images : les images d'un même patient se ressemblent,
    les traiter comme indépendantes donnerait un intervalle trop étroit.
    """
    rows_of = data.groupby("patient_id").indices
    patients = list(rows_of)
    scores = data["score"].to_numpy(dtype=float)
    values = data[metric].to_numpy(dtype=float)

    samples = []
    for _ in range(config.BOOTSTRAP_SAMPLES):
        drawn = rng.choice(len(patients), size=len(patients), replace=True)
        rows = np.concatenate([rows_of[patients[i]] for i in drawn])
        samples.append(correlation(scores[rows], values[rows], metric))
    if np.all(np.isnan(samples)):
        return np.nan, np.nan
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return float(low), float(high)


def intra_slice_correlation(data, metric):
    """Spearman moyen à anatomie égale : sCT de générateurs différents, même patient, même coupe."""
    values = [
        correlation(group["score"], group[metric], metric)
        for _, group in data.groupby(["patient_id", "slice_index"])
        if len(group) >= 3
    ]
    return float(np.nanmean(values)) if values else np.nan


def is_acceptable(values):
    values = np.asarray(values, dtype=float)
    if config.ACCEPTABLE_METRIC in config.LOWER_IS_BETTER:
        return values < config.ACCEPTABLE_THRESHOLD
    return values > config.ACCEPTABLE_THRESHOLD


def auroc(scores, positive):
    """Aire sous la courbe ROC : probabilité qu'un sCT acceptable ait une meilleure note qu'un autre."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = rankdata(scores)  # les égalités reçoivent le rang moyen, comme il se doit
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def filtering(data, metric):
    """Ne garder que la part KEEP_FRACTION des images les mieux notées : qu'y gagne-t-on ?

    On compare la métrique moyenne des images gardées à celle d'un filtrage au hasard (qui
    vaut, en moyenne, celle de toutes les images) et à celle d'un filtrage « oracle » qui
    garderait les vraies meilleures. gain_vs_oracle = 1 : aussi bien que l'oracle ;
    0 : pas mieux que le hasard ; négatif : pire que le hasard.
    """
    values = data[metric].to_numpy(dtype=float)
    scores = data["score"].to_numpy(dtype=float)
    n_keep = max(1, int(round(config.KEEP_FRACTION * len(values))))

    # On garde toutes les images notées au-dessus de la note seuil. Les images ex aequo sur
    # le seuil se partagent les places restantes : chacune compte pour une fraction, ce qui
    # donne exactement la moyenne d'un tirage au hasard entre elles.
    cutoff = np.sort(scores)[::-1][n_keep - 1]
    above, tied = scores > cutoff, scores == cutoff
    weights = above.astype(float)
    weights[tied] = (n_keep - above.sum()) / tied.sum()
    best_first = np.sort(values) if metric in config.LOWER_IS_BETTER else np.sort(values)[::-1]

    random_mean = values.mean()
    judge_mean = (weights * values).sum() / n_keep
    oracle_mean = best_first[:n_keep].mean()
    gain = (random_mean - judge_mean) / (random_mean - oracle_mean) if random_mean != oracle_mean else np.nan
    return {
        "n_images": len(values),
        "n_kept": n_keep,
        f"{metric}_random": round(random_mean, 3),
        f"{metric}_judge": round(judge_mean, 3),
        f"{metric}_oracle": round(oracle_mean, 3),
        "gain_vs_oracle": round(gain, 3),
    }


# --- Tableaux -----------------------------------------------------------------------

def judge_tables(merged, failed, n_testbench, rng):
    """Calcule leaderboard, corrélations et filtrage pour chaque (modèle, prompt)."""
    target = config.TARGET_METRIC
    leaderboard, correlations, filters = [], [], []

    for (model, prompt_name), data in merged.groupby(JUDGE_KEY, sort=False):
        judge = {"model": model, "prompt_name": prompt_name}
        n_errors = len(failed[(failed["model"] == model) & (failed["prompt_name"] == prompt_name)])
        low, high = bootstrap_interval(data, target, rng)

        leaderboard.append({
            **judge,
            "n_images": len(data),
            "coverage": round(len(data) / n_testbench, 3),
            "n_errors": n_errors,
            "score_mean": round(data["score"].mean(), 1),
            "score_std": round(data["score"].std(ddof=0), 1),
            f"spearman_{target}": round(correlation(data["score"], data[target], target), 3),
            f"spearman_{target}_ci_low": round(low, 3),
            f"spearman_{target}_ci_high": round(high, 3),
            f"pearson_{target}": round(correlation(data["score"], data[target], target, pearsonr), 3),
            "spearman_intra": round(intra_slice_correlation(data, target), 3),
            "auroc": round(auroc(data["score"], is_acceptable(data[config.ACCEPTABLE_METRIC])), 3),
            "latency_s_mean": round(data["latency_s"].mean(), 2),
            "vram_gb": round(data["vram_gb"].max(), 1),
        })
        correlations.append({
            **judge,
            **{f"spearman_{m}": round(correlation(data["score"], data[m], m), 3) for m in METRICS if m in data},
        })
        filters.append({**judge, **filtering(data, target)})

    # Un modèle dont toutes les réponses ont échoué apparaît quand même, en bas du classement.
    judged = set(map(tuple, merged[JUDGE_KEY].drop_duplicates().values))
    for (model, prompt_name), errors in failed.groupby(JUDGE_KEY, sort=False):
        if (model, prompt_name) not in judged:
            leaderboard.append({"model": model, "prompt_name": prompt_name, "n_images": 0,
                                "coverage": 0.0, "n_errors": len(errors)})

    leaderboard = pd.DataFrame(leaderboard).sort_values(f"spearman_{target}", ascending=False)
    return leaderboard, pd.DataFrame(correlations), pd.DataFrame(filters)


def generator_table(testbench):
    """Qualité de chaque générateur, sur le volume entier (une ligne par sCT, pas par coupe)."""
    volumes = testbench.drop_duplicates(["patient_id", "generator"])
    table = volumes.groupby("generator").agg(
        n_patients=("patient_id", "nunique"),
        mae_hu_mean=("mae_hu", "mean"),
        mae_hu_median=("mae_hu", "median"),
        psnr_db_mean=("psnr_db", "mean"),
        ssim_mean=("ssim", "mean"),
        n_maybe_seen=("maybe_seen_in_training", "sum"),
        **{f"{m}_mean": (m, "mean") for m in METRICS if m not in IMAGE_METRICS},
    )
    # MAE par centre : utile pour repérer un générateur qui a déjà vu les patients d'un centre.
    by_center = volumes.pivot_table(index="generator", columns="center", values="mae_hu", aggfunc="mean")
    by_center.columns = [f"mae_hu_center_{center}" for center in by_center.columns]
    return table.join(by_center).round(3).reset_index()


# --- Figure -------------------------------------------------------------------------

def scatter_plot(merged, leaderboard, path):
    """Un nuage de points par (modèle, prompt) : note du VLM en fonction de la métrique de référence."""
    target = config.TARGET_METRIC
    judges = leaderboard.loc[leaderboard["n_images"] > 0, JUDGE_KEY].values.tolist()  # ordre du leaderboard
    several_prompts = merged["prompt_name"].nunique() > 1

    columns = min(4, len(judges))
    lines = int(np.ceil(len(judges) / columns))
    figure, axes = plt.subplots(lines, columns, figsize=(3.6 * columns, 3.2 * lines),
                                sharex=True, sharey=True, squeeze=False)
    rho_column = f"spearman_{target}"

    for ax, (model, prompt_name) in zip(axes.flat, judges):
        data = merged[(merged["model"] == model) & (merged["prompt_name"] == prompt_name)]
        rho = leaderboard.loc[(leaderboard["model"] == model) & (leaderboard["prompt_name"] == prompt_name), rho_column].iloc[0]
        ax.scatter(data[target], data["score"], s=22, color="#2a78d6", alpha=0.75,
                   edgecolors="white", linewidths=0.6)
        title = f"{model} · {prompt_name}" if several_prompts else model
        ax.set_title(f"{title}\nρ orienté = {rho:.2f}", fontsize=9, color="#0b0b0b")
        ax.set_ylim(-3, 103)
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b5b4ae")
        ax.tick_params(colors="#52514e", labelsize=8)

    for ax in axes.flat[len(judges):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(METRIC_LABELS.get(target, target), fontsize=9, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("Note du VLM (0 à 100)", fontsize=9, color="#52514e")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# --- Programme principal ------------------------------------------------------------

def default_output_dir(testbench, testbench_csv):
    splits, styles = testbench["split"].unique(), testbench["render_style"].unique()
    if len(splits) == 1 and len(styles) == 1:
        return config.ANALYSIS_DIR / f"{splits[0]}_{styles[0]}"
    return config.ANALYSIS_DIR / f"{testbench_csv.parent.parent.name}_{testbench_csv.parent.name}"


def main():
    parser = argparse.ArgumentParser(description="Corrélations entre notes des VLM et métriques : leaderboard.")
    parser.add_argument("--testbench", required=True, help="testbench.csv produit par make_testbench.py")
    parser.add_argument("--judgements", default=config.DEFAULT_RESULTS_CSV, help="CSV produit par benchmark.py")
    parser.add_argument("--outdir", help="dossier de sortie (par défaut : results/analysis/<split>_<style>)")
    args = parser.parse_args()

    testbench_csv = Path(args.testbench)
    if not testbench_csv.exists():
        raise SystemExit(f"{testbench_csv} introuvable : lancer d'abord make_testbench.py.")
    testbench = pd.read_csv(testbench_csv)
    testbench, dose_columns = add_dose_metrics(testbench, testbench_csv)
    METRICS.extend(dose_columns)
    ok, failed = load_judgements(Path(args.judgements))

    # Seules les notes des images de CE banc d'essai sont gardées (pas celles d'un autre split
    # ou d'un autre style de rendu, qui sont dans le même judgements.csv).
    merged = ok.merge(testbench, on="item_id", how="inner")
    failed = failed[failed["item_id"].isin(testbench["item_id"])]
    if merged.empty:
        raise SystemExit("Aucune note ne correspond à ce banc d'essai : vérifier --testbench et --judgements.")

    output_dir = Path(args.outdir) if args.outdir else default_output_dir(testbench, testbench_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    acceptable = is_acceptable(testbench[config.ACCEPTABLE_METRIC])
    print(f"{len(merged)} notes rattachées à {len(testbench)} images ({testbench['patient_id'].nunique()} patients).")
    print(f"sCT acceptables ({config.ACCEPTABLE_METRIC} {'<' if config.ACCEPTABLE_METRIC in config.LOWER_IS_BETTER else '>'} "
          f"{config.ACCEPTABLE_THRESHOLD}) : {acceptable.sum()} / {len(testbench)} images.")
    if acceptable.all() or not acceptable.any():
        print("  Attention : une seule classe, l'AUROC n'est pas calculable (ajuster ACCEPTABLE_THRESHOLD).")

    rng = np.random.default_rng(config.SEED)
    leaderboard, correlations, filters = judge_tables(merged, failed, len(testbench), rng)

    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    correlations.to_csv(output_dir / "correlations.csv", index=False)
    filters.to_csv(output_dir / "filtering.csv", index=False)
    generator_table(testbench).to_csv(output_dir / "generators.csv", index=False)
    merged.to_csv(output_dir / "scores_and_metrics.csv", index=False)
    scatter_plot(merged, leaderboard, output_dir / "scatter.png")

    target = config.TARGET_METRIC
    shown = ["model", "prompt_name", "n_images", f"spearman_{target}", f"spearman_{target}_ci_low",
             f"spearman_{target}_ci_high", "spearman_intra", "auroc", "latency_s_mean", "vram_gb"]
    print(f"\nLeaderboard (métrique de référence : {target}, corrélations orientées) :")
    print(leaderboard[shown].to_string(index=False))
    print(f"\nRésultats dans {output_dir}")


if __name__ == "__main__":
    main()
