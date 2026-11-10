"""Charger, décharger et inspecter les VLM servis par Ollama.

S'utilise depuis Python :

    from vlm_manager import get_client, load_model, unload_model
    client = get_client()
    load_model(client, "medgemma:4b")
    ...
    unload_model(client, "medgemma:4b")

ou en ligne de commande :

    python vlm_manager.py list                  # modèles installés
    python vlm_manager.py ps                    # modèles en mémoire et VRAM occupée
    python vlm_manager.py pull medgemma:4b      # télécharger un modèle
    python vlm_manager.py load medgemma:4b      # charger en VRAM
    python vlm_manager.py unload medgemma:4b    # décharger
    python vlm_manager.py info                  # version exacte de chaque modèle, dans results/ollama_models.csv

La station et son GPU sont partagés : on ne décharge que les modèles qu'on a soi-même
chargés, jamais celui de quelqu'un d'autre (vérifier avec `ps`).
"""

import argparse
import csv
import time
from datetime import datetime

import ollama

import config


def get_client():
    """Crée un client connecté au serveur Ollama défini dans config.py."""
    return ollama.Client(host=config.OLLAMA_HOST, timeout=config.REQUEST_TIMEOUT_S)


def full_name(model):
    """Ollama ajoute le tag ':latest' quand on n'en précise pas ('moondream' -> 'moondream:latest').

    On fait pareil pour pouvoir comparer nos noms avec ceux que renvoie Ollama.
    """
    return model if ":" in model else f"{model}:latest"


def check_server(client):
    """Vérifie que le serveur Ollama répond, avec un message clair sinon."""
    try:
        client.list()
    except ConnectionError:
        raise SystemExit(
            f"Impossible de joindre Ollama à l'adresse {config.OLLAMA_HOST}.\n"
            "Le script doit être lancé sur la station (Ollama n'accepte que les connexions locales).\n"
            "Si c'est le cas, le service redémarre peut-être : réessayer, ou prévenir l'administrateur."
        )


def installed_models(client):
    """Liste des modèles téléchargés sur le serveur."""
    return [m.model for m in client.list().models]


def loaded_models(client):
    """Modèles actuellement en mémoire, avec la VRAM occupée par chacun (en Go)."""
    return {m.model: m.size_vram / 1e9 for m in client.ps().models}


PROFILE_COLUMNS = ["model", "digest", "family", "parameter_size", "quantization", "capabilities", "recorded_at"]


def model_profile(client, model):
    """Version exacte et caractéristiques d'un modèle installé.

    Renvoie un dict : empreinte (digest, qui change si le tag est mis à jour), famille, nombre
    de paramètres, quantification, capacités (« vision », « thinking »...) et thinking (bool).
    """
    model = full_name(model)
    profile = dict.fromkeys(PROFILE_COLUMNS, "")
    profile["model"] = model
    details = None
    for installed in client.list().models:
        if installed.model == model:
            profile["digest"] = installed.digest or ""
            details = installed.details
    capabilities = []
    try:
        shown = client.show(model)
        capabilities = list(shown.capabilities or [])
        details = shown.details or details
    except ollama.ResponseError:
        pass  # modèle absent : on renvoie ce qu'on a
    if details is not None:
        profile["family"] = details.family or ""
        profile["parameter_size"] = details.parameter_size or ""
        profile["quantization"] = details.quantization_level or ""
    profile["capabilities"] = " ".join(capabilities)
    profile["recorded_at"] = datetime.now().isoformat(timespec="seconds")
    profile["thinking"] = "thinking" in capabilities
    return profile


def save_profiles(profiles, path=None):
    """Met à jour le CSV des versions de modèles (une ligne par modèle, la plus récente)."""
    path = path or config.MODELS_CSV
    rows = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = {row["model"]: row for row in csv.DictReader(f)}
    for profile in profiles:
        rows[profile["model"]] = {column: profile.get(column, "") for column in PROFILE_COLUMNS}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))


def pull_model(client, model):
    """Télécharge un modèle en affichant la progression."""
    print(f"Téléchargement de {model}...")
    for progress in client.pull(model, stream=True):
        if progress.total:
            percent = 100 * (progress.completed or 0) / progress.total
            print(f"\r  {progress.status} : {percent:5.1f} %", end="", flush=True)
    print(f"\n{model} est installé.")


def load_model(client, model):
    """Charge un modèle en VRAM et renvoie (temps de chargement en s, VRAM occupée en Go).

    Envoyer un prompt vide suffit : Ollama charge le modèle sans rien générer.
    keep_alive=-1 le garde en mémoire jusqu'à ce qu'on le décharge nous-mêmes
    (sinon Ollama le décharge de lui-même après 5 minutes d'inactivité).
    """
    model = full_name(model)
    start = time.perf_counter()
    client.generate(model=model, prompt="", keep_alive=-1)
    load_time = time.perf_counter() - start

    for running in client.ps().models:
        if running.model == model:
            # Si le modèle ne tient pas entièrement sur le GPU, Ollama en met une partie
            # en RAM : ça marche, mais c'est beaucoup plus lent et ça fausse les latences.
            if running.size_vram < running.size:
                cpu_part = (running.size - running.size_vram) / 1e9
                print(f"  Attention : {cpu_part:.1f} Go de {model} sont en RAM (CPU), pas en VRAM.")
            return load_time, running.size_vram / 1e9

    return load_time, 0.0


def unload_model(client, model, wait_s=30):
    """Décharge un modèle de la VRAM et attend que la mémoire soit réellement libérée.

    keep_alive=0 demande à Ollama de décharger le modèle tout de suite. La libération
    prend parfois une ou deux secondes : on attend pour ne pas charger le modèle
    suivant alors que la VRAM est encore occupée.
    """
    model = full_name(model)
    client.generate(model=model, prompt="", keep_alive=0)

    deadline = time.time() + wait_s
    while model in loaded_models(client):
        if time.time() > deadline:
            print(f"  Attention : {model} est toujours en mémoire après {wait_s} s.")
            return
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Gestion des VLM Ollama.")
    parser.add_argument("action", choices=["list", "ps", "pull", "load", "unload", "info"])
    parser.add_argument("model", nargs="?", help="nom du modèle, ex. medgemma:4b")
    args = parser.parse_args()

    if args.action in ("pull", "load", "unload") and not args.model:
        parser.error(f"l'action '{args.action}' demande un nom de modèle")

    client = get_client()
    check_server(client)

    if args.action == "list":
        for model in installed_models(client):
            print(model)

    elif args.action == "ps":
        loaded = loaded_models(client)
        if not loaded:
            print("Aucun modèle en mémoire.")
        for model, vram_gb in loaded.items():
            print(f"{model:<25} {vram_gb:5.1f} Go de VRAM")

    elif args.action == "pull":
        pull_model(client, args.model)

    elif args.action == "load":
        load_time, vram_gb = load_model(client, args.model)
        print(f"{args.model} chargé en {load_time:.1f} s ({vram_gb:.1f} Go de VRAM).")

    elif args.action == "unload":
        unload_model(client, args.model)
        print(f"{args.model} déchargé.")

    elif args.action == "info":
        models = [args.model] if args.model else installed_models(client)
        profiles = [model_profile(client, model) for model in models]
        for p in profiles:
            print(f"{p['model']:<22} {p['parameter_size']:>7} {p['quantization']:<8} {p['digest'][:12]}  {p['capabilities']}")
        save_profiles(profiles)
        print(f"Écrit dans {config.MODELS_CSV}")


if __name__ == "__main__":
    main()
