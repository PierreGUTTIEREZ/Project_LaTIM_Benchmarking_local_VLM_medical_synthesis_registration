"""Le modèle de synthèse : un U-Net 2D qui prédit une coupe CT à partir d'une coupe IRM."""

import numpy as np
import torch
from monai.networks.nets import BasicUNet

import config


def build_model():
    return BasicUNet(spatial_dims=2, in_channels=1, out_channels=1, features=config.UNET_FEATURES)


def load_checkpoint(path, device):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


@torch.no_grad()
def predict_volume(model, source, device, batch_size=32):
    """Prédit le CT d'un volume entier, coupe axiale par coupe axiale.

    `source` est le volume normalisé de forme (z, y, x). Le résultat a la même forme,
    avec des valeurs dans [0, 1] (voir data.to_hu pour repasser en HU).
    """
    model.eval()
    depth, height, width = source.shape

    # Le U-Net divise la taille de l'image par deux à quatre reprises : on complète chaque
    # coupe jusqu'à un multiple de 16, et on retire ce bord après la prédiction.
    pad_h = -height % 16
    pad_w = -width % 16

    prediction = np.zeros((depth, height, width), dtype=np.float32)
    for start in range(0, depth, batch_size):
        batch = np.asarray(source[start:start + batch_size], dtype=np.float32)
        batch = np.pad(batch, ((0, 0), (0, pad_h), (0, pad_w)))
        output = model(torch.from_numpy(batch)[:, None].to(device))
        prediction[start:start + batch_size] = output[:, 0, :height, :width].clamp(0, 1).cpu().numpy()

    return prediction
