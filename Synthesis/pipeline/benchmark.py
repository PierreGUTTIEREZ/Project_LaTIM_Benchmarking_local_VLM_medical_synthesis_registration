"""Banc d'essai : chaque VLM note chaque image, et tout est enregistré dans un CSV.

Pour chaque modèle : on le charge, on lui fait noter toutes les images, puis on le
décharge avant de passer au suivant.

Le GPU de latim5090 est partagé : on ne décharge jamais le modèle de quelqu'un d'autre.
Si un autre modèle est en mémoire, un avertissement s'affiche ; Ollama libérera la place
si besoin, mais les temps mesurés peuvent en être faussés. Mieux vaut lancer les longs
benchmarks aux heures creuses.

Format du CSV du banc d'essai (produit par sct_generation/make_testbench.py) :

    item_id,image_path,...
    1BA001_unet_baseline-epoch_005_z072_gray,renders/1BA001_unet_baseline-epoch_005_z072_gray.png,...
    1BA001_fedsynth-li_z072_gray,renders/1BA001_fedsynth-li_z072_gray.png,...

    item_id     identifiant unique de l'image
    image_path  chemin du PNG, absolu ou relatif au dossier du CSV

Le fichier contient d'autres colonnes (MAE, PSNR, SSIM...) : elles sont ignorées ici,
car le VLM ne doit jamais voir les métriques. On rattache les scores aux métriques plus tard,
dans analyze.py, grâce à item_id.

Le CSV de résultats est écrit au fur et à mesure. Si le script s'arrête en route, on le relance
avec le même fichier de sortie et les jugements déjà réussis sont sautés. Les lignes en erreur
(et les notes sans justification) sont retentées ; leurs anciennes versions restent dans le
fichier et seront filtrées à l'analyse (colonne `error` non vide).

Pendant le benchmark, une image en erreur (réponse coupée, erreur 500 « token repeat limit »...)
est retentée tout de suite, jusqu'à MAX_ATTEMPTS fois (voir RETRY_STRATEGIES dans config.py).
La colonne `attempt` dit à quel essai la note a été obtenue, `retry_errors` garde les erreurs
des essais ratés. La version exacte de chaque modèle est notée dans results/ollama_models.csv.

Exemples :

    uv run python benchmark.py --testbench chemin/testbench.csv --models medgemma:latest
    uv run python benchmark.py --testbench chemin/testbench.csv            # modèles installés seulement
    uv run python benchmark.py --testbench chemin/testbench.csv --pull     # télécharge les manquants
"""

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import config
from judge import judge_with_retries, token_budget
from vlm_manager import (
    check_server,
    full_name,
    get_client,
    installed_models,
    load_model,
    loaded_models,
    model_profile,
    pull_model,
    save_profiles,
    unload_model,
)

RESULT_COLUMNS = [
    "model",
    "prompt_name",
    "item_id",
    "score",
    "reason",
    "latency_s",
    "vram_gb",
    "error",
    "timestamp",
    "attempt",         # essai qui a donné la note (1 = protocole de base)
    "tokens",          # tokens générés, réflexion comprise
    "thinking_chars",  # longueur de la réflexion cachée (0 si le modèle ne réfléchit pas)
    "retry_errors",    # erreurs des essais ratés, séparées par « | »
    "model_digest",    # empreinte du modèle (voir results/ollama_models.csv)
]


def read_testbench(testbench_csv):
    """Lit le CSV du banc d'essai et renvoie une liste de {item_id, image_path}."""
    testbench_csv = Path(testbench_csv)
    items = []
    missing = []

    with open(testbench_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = testbench_csv.parent / image_path
            if not image_path.exists():
                missing.append(image_path)
            items.append({"item_id": row["item_id"], "image_path": image_path})

    # Mieux vaut s'en rendre compte tout de suite qu'au bout de deux heures de benchmark.
    if missing:
        raise SystemExit(f"{len(missing)} image(s) introuvable(s), par exemple : {missing[0]}")

    return items


def upgrade_results_csv(results_csv):
    """Ajoute au CSV de résultats les colonnes apparues depuis sa création (vides pour les anciennes lignes).

    Sans ça, les nouvelles lignes seraient écrites avec plus de colonnes que l'en-tête.
    Une copie de l'ancien fichier est gardée à côté, au cas où.
    """
    if not results_csv.exists():
        return
    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == RESULT_COLUMNS:
            return
        rows = list(reader)
    backup = results_csv.with_name(f"{results_csv.stem}_avant_{datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy2(results_csv, backup)
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
    print(f"{results_csv.name} : colonnes ajoutées ({len(rows)} lignes). Ancienne version gardée dans {backup.name}.")


def already_done(results_csv):
    """Renvoie l'ensemble des (model, prompt_name, item_id) déjà jugés sans erreur et avec une justification."""
    if not results_csv.exists():
        return set()

    with open(results_csv, newline="", encoding="utf-8") as f:
        return {
            (row["model"], row["prompt_name"], row["item_id"])
            for row in csv.DictReader(f)
            if not row["error"] and (row.get("reason") or "").strip()
        }


def append_result(results_csv, row):
    """Ajoute une ligne au CSV de résultats (et l'en-tête si le fichier est nouveau)."""
    is_new_file = not results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def run_model(client, model, items, prompt_name, results_csv):
    """Charge un modèle, lui fait noter toutes les images, puis le décharge."""
    others = [name for name in loaded_models(client) if name != model]
    if others:
        print(f"  Attention : {', '.join(others)} déjà en mémoire (quelqu'un l'utilise peut-être).")

    profile = model_profile(client, model)
    save_profiles([profile])
    budget = token_budget(model, profile)
    load_time, vram_gb = load_model(client, model)
    print(f"{model} chargé en {load_time:.1f} s ({vram_gb:.1f} Go de VRAM), "
          f"{profile['parameter_size'] or '?'} paramètres, réflexion : {'oui' if profile['thinking'] else 'non'}, "
          f"plafond {budget} tokens.")

    counts = {"premier essai": 0, "après nouvel essai": 0, "en échec": 0}
    try:
        for i, item in enumerate(items, start=1):
            row = {
                "model": model,
                "prompt_name": prompt_name,
                "item_id": item["item_id"],
                "vram_gb": round(vram_gb, 2),
                "error": "",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_digest": profile["digest"][:12],
            }
            # Une image qui pose problème (timeout, réponse invalide...) ne doit pas
            # arrêter tout le banc : on la retente, puis on note l'erreur et on passe à la suivante.
            result, attempt, errors = judge_with_retries(
                client, model, item["image_path"], config.PROMPTS[prompt_name], budget, profile["thinking"])
            row["attempt"] = attempt
            row["retry_errors"] = " | ".join(errors)
            if result:
                row.update(result)
                counts["premier essai" if attempt == 1 else "après nouvel essai"] += 1
                status = f"score {row['score']:3d}  ({row['latency_s']} s)" + (f"  essai {attempt}" if attempt > 1 else "")
            else:
                row["error"] = errors[-1]
                counts["en échec"] += 1
                status = f"ERREUR après {attempt} essais : {row['error']}"

            append_result(results_csv, row)
            print(f"  [{i}/{len(items)}] {item['item_id']} : {status}")
    finally:
        # Exécuté même après un Ctrl+C, pour ne pas laisser la VRAM occupée.
        unload_model(client, model)
        print(f"{model} déchargé. Notes : " + ", ".join(f"{n} {label}" for label, n in counts.items()) + "\n")


def run_benchmark(testbench_csv, models, prompt_name, results_csv, pull_missing):
    client = get_client()
    check_server(client)

    items = read_testbench(testbench_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    upgrade_results_csv(results_csv)
    done = already_done(results_csv)
    installed = installed_models(client)

    print(f"{len(items)} images, {len(models)} modèles, prompt '{prompt_name}'.")
    print(f"Résultats : {results_csv}\n")

    failed = []
    for model in models:
        model = full_name(model)

        todo = [item for item in items if (model, prompt_name, item["item_id"]) not in done]
        if not todo:
            print(f"{model} : tout est déjà jugé.\n")
            continue

        if model not in installed:
            if not pull_missing:
                print(f"{model} n'est pas installé, on passe (relancer avec --pull pour le télécharger).\n")
                continue

        # Un modèle qui ne se télécharge pas ou ne se charge pas (nom inconnu, pas assez de
        # VRAM...) ne doit pas empêcher de tester les suivants. Ctrl+C arrête tout, lui.
        try:
            if model not in installed:
                pull_model(client, model)
            run_model(client, model, todo, prompt_name, results_csv)
        except Exception as error:
            failed.append(model)
            print(f"{model} : ERREUR {type(error).__name__}: {error}\nOn passe au modèle suivant.\n")

    if failed:
        print(f"Modèles en échec : {', '.join(failed)}. Relancer la même commande pour réessayer.")
    print("Benchmark terminé.")


def main():
    parser = argparse.ArgumentParser(description="Faire noter le banc d'essai par plusieurs VLM.")
    parser.add_argument("--testbench", default=config.TESTBENCH_CSV, help="CSV du banc d'essai")
    parser.add_argument("--models", nargs="+", default=config.MODELS, help="modèles à évaluer")
    parser.add_argument("--prompt", default=config.DEFAULT_PROMPT, choices=config.PROMPTS)
    parser.add_argument("--output", default=config.DEFAULT_RESULTS_CSV, help="CSV de résultats")
    parser.add_argument("--pull", action="store_true", help="télécharger les modèles pas encore installés")
    args = parser.parse_args()

    if args.testbench is None:
        parser.error("indiquer le banc d'essai avec --testbench (ou TESTBENCH_CSV dans config.py)")

    run_benchmark(args.testbench, args.models, args.prompt, Path(args.output), args.pull)


if __name__ == "__main__":
    main()
