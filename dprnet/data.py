"""Dataset discovery, pre-processing and patch sampling for DRIVE and CHASE_DB1.

Expected layouts (the official archives, unpacked unchanged):

DRIVE/
  training/images/21_training.tif   training/1st_manual/21_manual1.gif   training/mask/21_training_mask.gif
  test/images/01_test.tif           test/1st_manual/01_manual1.gif       test/mask/01_test_mask.gif

CHASE_DB1/
  Image_01L.jpg  Image_01L_1stHO.png  Image_01L_2ndHO.png  ...  Image_14R.jpg
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from .geometry import soft_skeleton, normalised_radius

# Images 22, 27, 34 and 39 of the DRIVE training set are held out for validation.
DRIVE_VALIDATION = ("22", "27", "34", "39")
# Images 1-20 of CHASE_DB1 (01L ... 10R) are used for training, 11L ... 14R for testing.
CHASE_VALIDATION = ("03L", "05R", "08L", "10R")


@dataclass
class Sample:
    name: str
    image: Path
    label: Path
    fov: Path | None = None       # None -> estimated from the image (CHASE_DB1)
    label2: Path | None = None    # second observer, if available


def _check(samples, expected, where):
    if len(samples) != expected:
        raise FileNotFoundError(f"Expected {expected} images under {where}, found {len(samples)}")
    for s in samples:
        for p in (s.image, s.label, s.fov):
            if p is not None and not p.is_file():
                raise FileNotFoundError(p)
    return samples


def discover(dataset: str, root: Path, split: str) -> list[Sample]:
    """Return the samples of ``split`` in {'train', 'val', 'trainval', 'test'}."""
    root = Path(root)
    if dataset == "drive":
        folder = root / ("test" if split == "test" else "training")
        samples = []
        for img in sorted((folder / "images").glob("*.tif")):
            cid = img.name.split("_")[0]
            samples.append(Sample(cid, img, folder / "1st_manual" / f"{cid}_manual1.gif",
                                  folder / "mask" / img.name.replace(".tif", "_mask.gif"),
                                  folder / "2nd_manual" / f"{cid}_manual2.gif"))
        _check(samples, 20, folder)
        held = DRIVE_VALIDATION
    elif dataset == "chase":
        samples = []
        for img in sorted(root.glob("Image_*.jpg")):
            cid = img.stem.split("_")[1]
            samples.append(Sample(cid, img, root / f"{img.stem}_1stHO.png", None,
                                  root / f"{img.stem}_2ndHO.png"))
        _check(samples, 28, root)
        samples = samples[20:] if split == "test" else samples[:20]
        held = CHASE_VALIDATION
    else:
        raise ValueError(dataset)
    for s in samples:
        if s.label2 is not None and not s.label2.is_file():
            s.label2 = None
    if split == "train":
        return [s for s in samples if s.name not in held]
    if split == "val":
        return [s for s in samples if s.name in held]
    return samples


def estimate_fov(image: np.ndarray) -> np.ndarray:
    """FOV mask by thresholding the red channel, then filling holes and keeping the largest region."""
    red = cv2.GaussianBlur(image[..., 0], (5, 5), 0)
    mask = red > 20
    mask = ndimage.binary_fill_holes(ndimage.binary_opening(mask, iterations=3))
    lab, n = ndimage.label(mask)
    if n > 1:
        mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndimage.binary_erosion(mask, iterations=2)


def load(sample: Sample):
    """Return (RGB uint8 image, binary label, binary FOV) as numpy arrays."""
    image = np.asarray(Image.open(sample.image).convert("RGB"), dtype=np.uint8)
    label = np.asarray(Image.open(sample.label).convert("L")) > 127
    fov = (np.asarray(Image.open(sample.fov).convert("L")) > 127) if sample.fov else estimate_fov(image)
    if image.shape[:2] != label.shape or label.shape != fov.shape:
        raise ValueError(f"Image, label and FOV sizes differ for {sample.image}")
    return image, label.astype(np.float32), fov.astype(np.float32)


def features(image: np.ndarray) -> np.ndarray:
    """RGB plus CLAHE-enhanced green channel, scaled to [-1, 1]; returns H x W x 4 float32."""
    green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image[..., 1])
    x = np.concatenate([image, green[..., None]], axis=2).astype(np.float32) / 255.0
    return x * 2.0 - 1.0


class PatchDataset(Dataset):
    """Random 128 x 128 patches centred near skeleton, vessel, FOV or thin-centreline points.

    ``thin_share`` > 0 (phase two) draws that fraction of the centres from the centrelines
    of thin vessels (normalised radius <= 2).
    """

    def __init__(self, samples, dataset: str, length: int, patch: int = 128, seed: int = 73,
                 thin_share: float = 0.0):
        self.rng = random.Random(seed)
        self.length, self.patch, self.thin_share = length, patch, thin_share
        self.items = []
        for s in samples:
            image, label, fov = load(s)
            with torch.no_grad():
                skel = soft_skeleton(torch.from_numpy(label)[None, None])[0, 0].numpy()
            valid = fov > 0
            centre = (skel > 0.1) & valid
            radius = normalised_radius(label > 0, dataset)
            thin = centre & (radius <= 2.0)
            pts = [np.argwhere(centre), np.argwhere((label > 0) & valid), np.argwhere(valid), np.argwhere(thin)]
            self.items.append((features(image), label, fov, skel, pts))

    def __len__(self):
        return self.length

    def _centre(self, pts):
        r = self.rng.random()
        if self.thin_share and r < self.thin_share and len(pts[3]):
            p = pts[3]
        else:
            r = self.rng.random()
            p = pts[0] if r < 0.35 else pts[1] if r < 0.70 else pts[2]
        p = p if len(p) else pts[2]
        return p[self.rng.randrange(len(p))]

    def __getitem__(self, _):
        img, label, fov, skel, pts = self.rng.choice(self.items)
        h, w = label.shape
        cy, cx = self._centre(pts)
        top = min(max(int(cy) - self.rng.randrange(self.patch), 0), h - self.patch)
        left = min(max(int(cx) - self.rng.randrange(self.patch), 0), w - self.patch)
        arr = [a[top:top + self.patch, left:left + self.patch] for a in (img, label, fov, skel)]
        k = self.rng.randrange(4)
        arr = [np.rot90(a, k) for a in arr]
        if self.rng.random() < 0.5:
            arr = [a[:, ::-1] for a in arr]
        if self.rng.random() < 0.5:
            arr = [a[::-1] for a in arr]
        arr[0] = np.clip(arr[0] * self.rng.uniform(0.85, 1.15) + self.rng.uniform(-0.08, 0.08), -1, 1)
        x = torch.from_numpy(np.ascontiguousarray(arr[0].transpose(2, 0, 1))).float()
        y, v, s = (torch.from_numpy(np.ascontiguousarray(a))[None].float() for a in arr[1:])
        return x, y, v, s
