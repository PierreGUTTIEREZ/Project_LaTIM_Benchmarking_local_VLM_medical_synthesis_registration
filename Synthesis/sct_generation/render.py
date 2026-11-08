"""Rendu d'une coupe de CT (en HU) en image PNG pour les VLM.

L'image ne contient que la coupe : aucun texte, aucune légende, aucune information sur le
modèle qui l'a produite ni sur sa qualité.
"""

import numpy as np
from PIL import Image


def window(slice_hu, hu_min, hu_max):
    """HU -> niveaux de gris 0-255 : sous hu_min c'est noir, au-dessus de hu_max c'est blanc."""
    scaled = (np.clip(slice_hu, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
    return np.round(scaled * 255).astype(np.uint8)


def body_box(mask_slice, margin):
    """Boîte carrée (haut, bas, gauche, droite) qui contient le corps, avec une marge en pixels.

    La boîte peut déborder de l'image : crop() complète alors avec de l'air.
    """
    height, width = mask_slice.shape
    rows = np.flatnonzero(mask_slice.any(axis=1))
    cols = np.flatnonzero(mask_slice.any(axis=0))
    if rows.size == 0:
        # Pas de corps sur cette coupe : on garde l'image entière, rendue carrée.
        rows, cols = np.array([0, height - 1]), np.array([0, width - 1])

    size = max(rows[-1] - rows[0], cols[-1] - cols[0]) + 1 + 2 * margin
    top = (rows[0] + rows[-1] + 1) // 2 - size // 2
    left = (cols[0] + cols[-1] + 1) // 2 - size // 2
    return top, top + size, left, left + size


def crop(slice_hu, box, fill):
    """Découpe la boîte dans la coupe, en complétant avec `fill` si elle déborde."""
    top, bottom, left, right = box
    height, width = slice_hu.shape
    pad = max(0, -top, -left, bottom - height, right - width)
    padded = np.pad(slice_hu, pad, constant_values=fill)
    return padded[top + pad:bottom + pad, left + pad:right + pad]


def render_slice(slice_hu, style):
    """Coupe en HU (déjà recadrée, carrée) -> image PIL de style['size'] × style['size'] pixels."""
    windows = style["windows"]
    if len(windows) not in (1, 3):
        raise ValueError(f"Un style de rendu doit avoir 1 ou 3 fenêtres, pas {len(windows)}.")

    channels = [window(slice_hu, hu_min, hu_max) for hu_min, hu_max in windows]
    array = channels[0] if len(channels) == 1 else np.stack(channels, axis=-1)
    image = Image.fromarray(array)
    return image.resize((style["size"], style["size"]), Image.Resampling.LANCZOS)
