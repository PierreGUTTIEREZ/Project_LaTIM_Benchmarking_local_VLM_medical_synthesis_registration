"""Faire noter une image par un VLM.

Le VLM renvoie un score de 0 à 100 et une justification. Le format de la réponse
est imposé par un schéma JSON passé à Ollama (paramètre `format`) : le décodage est
contraint, donc même les petits modèles bavards renvoient un JSON valide.

Test rapide sur une image :

    python judge.py moondream chemin/vers/image.png
"""

import argparse
import time

from pydantic import BaseModel, Field

import config
from vlm_manager import check_server, full_name, get_client, loaded_models, unload_model


class Judgement(BaseModel):
    """Réponse attendue du VLM. Pydantic en déduit le schéma JSON et vérifie la réponse."""

    score: int = Field(ge=0, le=100)
    reason: str


def judge_image(client, model, image_path, prompt):
    """Demande à `model` de noter l'image et renvoie un dict {score, reason, latency_s}.

    Le modèle doit déjà être chargé (voir vlm_manager.load_model), sinon le premier
    appel inclut le temps de chargement et la latence mesurée n'a plus de sens.
    """
    start = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
        format=Judgement.model_json_schema(),
        options={"temperature": config.TEMPERATURE, "seed": config.SEED},
        keep_alive=-1,  # sans ça, Ollama remettrait le délai de déchargement à 5 minutes
    )
    latency_s = time.perf_counter() - start

    judgement = Judgement.model_validate_json(response.message.content)
    return {"score": judgement.score, "reason": judgement.reason, "latency_s": round(latency_s, 2)}


def main():
    parser = argparse.ArgumentParser(description="Faire noter une seule image par un VLM.")
    parser.add_argument("model", help="ex. moondream, medgemma:4b")
    parser.add_argument("image", help="chemin du PNG à juger")
    parser.add_argument("--prompt", default=config.DEFAULT_PROMPT, choices=config.PROMPTS)
    args = parser.parse_args()

    client = get_client()
    check_server(client)
    was_loaded = full_name(args.model) in loaded_models(client)

    result = judge_image(client, args.model, args.image, config.PROMPTS[args.prompt])
    print(f"Score  : {result['score']}/100")
    print(f"Raison : {result['reason']}")
    print(f"Temps  : {result['latency_s']} s (chargement du modèle compris s'il n'était pas en mémoire)")

    # judge_image garde le modèle en mémoire (keep_alive=-1). Pour un simple test, on rend
    # le GPU partagé dans l'état où on l'a trouvé.
    if not was_loaded:
        unload_model(client, args.model)


if __name__ == "__main__":
    main()
