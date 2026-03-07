import numpy as np
from PIL import Image

def load_mask(path):
    path = str(path)
    if path.endswith(".npy"):
        m = np.load(path)
        m = (m > 0).astype(np.uint8)
        return m
    # png/tif mask
    im = Image.open(path).convert("L")
    m = np.array(im)
    return (m > 0).astype(np.uint8)

def mask_to_level(mask, wsi, level: int):
    """
    Resize mask to match the dimensions of the given WSI level.
    Assumes mask corresponds to some level; for pilot, we treat it as already matching sample level
    if you saved it that way. If not, we’ll adapt once you tell me mask resolution.
    """
    # For v1: assume mask already matches sample level dims
    return mask