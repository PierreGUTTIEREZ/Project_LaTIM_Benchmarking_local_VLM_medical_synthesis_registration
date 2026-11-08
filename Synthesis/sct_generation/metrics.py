"""Métriques d'image entre un sCT et le vrai CT : MAE, PSNR et SSIM, dans le masque du corps.

On suit les définitions du challenge SynthRAD2023, pour que nos chiffres soient comparables
à ceux de la littérature :

  - les deux images sont d'abord ramenées dans la plage [CT_MIN_HU, CT_MAX_HU] = [-1024, 3000] ;
  - MAE  : moyenne de |sCT - CT| sur les voxels du corps, en HU (plus bas = meilleur) ;
  - PSNR : 10 log10(plage² / erreur quadratique moyenne dans le corps), en dB (plus haut = meilleur) ;
  - SSIM : similarité de structure, de 0 à 1 (plus haut = meilleur). Elle est calculée sur
           l'image entière, fond mis à -1024 HU, puis moyennée sur les voxels du corps.

Les mêmes fonctions marchent sur un volume 3D (z, y, x) comme sur une coupe 2D (y, x).
"""

import numpy as np
from skimage.metrics import structural_similarity

import config

DATA_RANGE = config.CT_MAX_HU - config.CT_MIN_HU


def _clip(image):
    return np.clip(np.asarray(image, dtype=np.float32), config.CT_MIN_HU, config.CT_MAX_HU)


def mae(sct, ct, mask):
    """Erreur absolue moyenne dans le corps, en HU."""
    body = mask > 0
    if not body.any():
        return float("nan")
    return float(np.abs(_clip(sct) - _clip(ct))[body].mean())


def psnr(sct, ct, mask):
    """Rapport signal sur bruit de crête dans le corps, en dB."""
    body = mask > 0
    if not body.any():
        return float("nan")
    mse = float(np.mean((_clip(sct) - _clip(ct))[body] ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(DATA_RANGE**2 / mse))


def ssim(sct, ct, mask):
    """SSIM moyenne dans le corps."""
    body = mask > 0
    if not body.any():
        return float("nan")
    # Fond à l'air pour les deux images, puis passage dans [0, 1].
    sct = (np.where(body, _clip(sct), config.CT_MIN_HU) - config.CT_MIN_HU) / DATA_RANGE
    ct = (np.where(body, _clip(ct), config.CT_MIN_HU) - config.CT_MIN_HU) / DATA_RANGE
    _, ssim_map = structural_similarity(ct, sct, data_range=1.0, full=True)
    return float(ssim_map[body].mean())


def all_metrics(sct, ct, mask, suffix=""):
    """Les trois métriques dans un dict : {'mae_hu', 'psnr_db', 'ssim'} (+ suffixe éventuel)."""
    return {
        f"mae_hu{suffix}": round(mae(sct, ct, mask), 2),
        f"psnr_db{suffix}": round(psnr(sct, ct, mask), 3),
        f"ssim{suffix}": round(ssim(sct, ct, mask), 4),
    }
