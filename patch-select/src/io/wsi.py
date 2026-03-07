import numpy as np
import openslide
from PIL import Image
import cv2

class WSI:
    def __init__(self, path: str):
        self.path = path
        self.osr = openslide.OpenSlide(path)

        # mpp can be missing sometimes; still okay for this half-mag rule
        self.mpp_x = float(self.osr.properties.get(openslide.PROPERTY_NAME_MPP_X, "nan"))
        self.mpp_y = float(self.osr.properties.get(openslide.PROPERTY_NAME_MPP_Y, "nan"))

        self.level_downsamples = list(self.osr.level_downsamples)
        self.level_dimensions = list(self.osr.level_dimensions)

    def close(self):
        self.osr.close()

    def read_region_rgb(self, x0: int, y0: int, level: int, size: int) -> np.ndarray:
        """
        Read region at (x0,y0) in level-0 reference frame.
        Returns RGB uint8 array (H,W,3).
        """
        im = self.osr.read_region((int(x0), int(y0)), int(level), (int(size), int(size)))
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)

    def read_half_mag_patch(self, x0: int, y0: int, out_size: int = 224, scale_factor: int = 2) -> np.ndarray:
        """
        Enforce half magnification by reading (out_size*scale_factor) at level 0
        and downsampling to out_size.
        """
        read_size = int(out_size * scale_factor)
        img = self.read_region_rgb(x0, y0, level=0, size=read_size)

        # Downsample with area interpolation (best for shrinking)
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return img

def open_wsi(path: str) -> WSI:
    return WSI(path)

def get_last_level(wsi: WSI) -> int:
    return len(wsi.level_dimensions) - 1