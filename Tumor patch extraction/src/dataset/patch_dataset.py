# dataset/patch_dataset.py
import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

try:
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError("torchvision is required. pip install torchvision") from e


def build_transforms(train: bool = True, img_size: int = 224):
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            # ImageNet normalization (good default for ResNet/ConvNeXt)
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


class PatchCSVDataset(Dataset):
    def __init__(self, csv_path: str, train: bool = True):
        self.df = pd.read_csv(csv_path)
        if "png_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"CSV must contain png_path and label. Got: {list(self.df.columns)}")

        self.transform = build_transforms(train=train)

        # sanity: keep only existing files
        exists = self.df["png_path"].apply(lambda p: os.path.exists(p))
        missing = (~exists).sum()
        if missing:
            self.df = self.df[exists].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["png_path"]
        y = int(row["label"])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)