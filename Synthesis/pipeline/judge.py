"""Faire noter une image par un VLM.

Le VLM renvoie un score de 0 à 100 et une justification. Le format de la réponse
est imposé par un schéma JSON passé à Ollama (paramètre `format`) : le décodage est
contraint, donc même les petits modèles bavards renvoient un JSON valide.

Certains modèles échouent quand même de temps en temps : réponse coupée au plafond de
tokens (boucle, ou réflexion trop longue) ou erreur 500 d'Ollama « token repeat limit
reached » (boucle détectée par le serveur). judge_with_retries() retente alors l'image
avec les réglages de RETRY_STRATEGIES (config.py), et dit à quel essai elle a réussi.

Test rapide sur une image :

    python judge.py moondream chemin/vers/image.png
"""

import argparse
import time
from pathlib import Path

import ollama
from pydantic import BaseModel, Field

import config
from vlm_manager import check_server, full_name, get_client, loaded_models, model_profile, unload_model


class Judgement(BaseModel):
    """Réponse attendue du VLM. Pydantic en déduit le schéma JSON et vérifie la réponse."""

    score: int = Field(ge=0, le=100)
    reason: str


def token_budget(model, profile=None):
    """Plafond de tokens d'un modèle : plus haut pour ceux qui réfléchissent avant de répondre."""
    if model in config.NUM_PREDICT_BY_MODEL:
        return config.NUM_PREDICT_BY_MODEL[model]
    if profile and profile.get("thinking"):
        return config.NUM_PREDICT_THINKING
    return config.NUM_PREDICT


def judge_image(client, model, image_path, prompt, num_predict=None, strategy=None, can_think=False):
    """Demande à `model` de noter l'image et renvoie un dict {score, reason, latency_s, tokens, thinking_chars}.

    `strategy` (un élément de RETRY_STRATEGIES) ajoute un message système, change des options
    ou coupe la réflexion ; sans elle, c'est le protocole de base (température 0, consigne seule).

    Le modèle doit déjà être chargé (voir vlm_manager.load_model), sinon le premier
    appel inclut le temps de chargement et la latence mesurée n'a plus de sens.
    """
    strategy = strategy or {}
    num_predict = num_predict or config.NUM_PREDICT
    options = {"temperature": config.TEMPERATURE, "seed": config.SEED, "num_predict": num_predict}
    options.update(strategy.get("options", {}))

    messages = []
    if strategy.get("system"):
        messages.append({"role": "system", "content": strategy["system"]})
    messages.append({"role": "user", "content": prompt, "images": [str(image_path)]})

    # Couper la réflexion n'a de sens que pour un modèle qui en a une : on ne passe `think`
    # qu'à ceux-là (sans `think`, chaque modèle garde son comportement par défaut).
    extra = {}
    if can_think and strategy.get("think") is not None:
        extra["think"] = strategy["think"]

    start = time.perf_counter()
    response = client.chat(
        model=model,
        messages=messages,
        format=Judgement.model_json_schema(),
        options=options,
        keep_alive=-1,  # sans ça, Ollama remettrait le délai de déchargement à 5 minutes
        **extra,
    )
    latency_s = time.perf_counter() - start
    thinking = response.message.thinking or ""
    content = response.message.content or ""

    if response.done_reason == "length":
        during = " pendant la réflexion" if thinking and not content.strip() else ""
        raise ValueError(f"réponse coupée après {num_predict} tokens{during} (voir NUM_PREDICT dans config.py)")
    judgement = Judgement.model_validate_json(content)
    # Le schéma accepte une chaîne vide : on la refuse, la consigne demande une justification.
    if not judgement.reason.strip():
        raise ValueError("justification vide")
    return {
        "score": judgement.score,
        "reason": judgement.reason,
        "latency_s": round(latency_s, 2),
        "tokens": response.eval_count,
        "thinking_chars": len(thinking),
    }


def judge_with_retries(client, model, image_path, prompt, num_predict=None, can_think=False):
    """Note l'image ; en cas d'échec, la retente avec RETRY_STRATEGIES.

    Renvoie (résultat ou None, numéro du dernier essai, liste des erreurs rencontrées).
    """
    strategies = [None, *config.RETRY_STRATEGIES][: max(1, config.MAX_ATTEMPTS)]
    errors = []
    for attempt, strategy in enumerate(strategies, start=1):
        try:
            result = judge_image(client, model, image_path, prompt, num_predict, strategy, can_think)
            return result, attempt, errors
        except Exception as error:  # Ctrl+C (KeyboardInterrupt) n'est pas attrapé : il arrête tout
            errors.append(f"{type(error).__name__}: {error}")
            if isinstance(error, (ollama.ResponseError, ConnectionError)) and attempt < len(strategies):
                time.sleep(config.RETRY_PAUSE_S)
    return None, len(strategies), errors


def main():
    parser = argparse.ArgumentParser(description="Faire noter une seule image par un VLM.")
    parser.add_argument("model", help="ex. moondream, medgemma:4b")
    parser.add_argument("image", help="chemin du PNG à juger")
    parser.add_argument("--prompt", default=config.DEFAULT_PROMPT, choices=config.PROMPTS)
    args = parser.parse_args()

    # Ollama prendrait un chemin inexistant pour une image encodée en base64 et renverrait
    # une erreur peu claire : on vérifie avant.
    if not Path(args.image).is_file():
        raise SystemExit(f"Image introuvable : {args.image}")

    model = full_name(args.model)
    client = get_client()
    check_server(client)
    was_loaded = model in loaded_models(client)
    profile = model_profile(client, model)

    result, attempt, errors = judge_with_retries(
        client, model, args.image, config.PROMPTS[args.prompt], token_budget(model, profile), profile["thinking"])
    for error in errors:
        print(f"Erreur : {error}")
    if result:
        print(f"Score  : {result['score']}/100 (essai {attempt})")
        print(f"Raison : {result['reason']}")
        print(f"Temps  : {result['latency_s']} s (chargement du modèle compris s'il n'était pas en mémoire)")
        print(f"Tokens : {result['tokens']} générés, dont une réflexion de {result['thinking_chars']} caractères")

    # judge_image garde le modèle en mémoire (keep_alive=-1). Pour un simple test, on rend
    # le GPU partagé dans l'état où on l'a trouvé.
    if not was_loaded:
        unload_model(client, model)


if __name__ == "__main__":
    main()
