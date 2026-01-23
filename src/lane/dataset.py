"""SDLane Dataset for lane semantic segmentation"""
import os
import glob
import json
import random
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resize_keep_aspect(
    H: int, W: int,
    shortest_edge: int,
    longest_edge: int
) -> Tuple[int, int]:
    """Resize keeping aspect ratio within bounds"""
    scale = shortest_edge / min(H, W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    if max(newH, newW) > longest_edge:
        scale = longest_edge / max(newH, newW)
        newH, newW = int(round(newH * scale)), int(round(newW * scale))
    return max(1, newH), max(1, newW)


def pad_to_divisor(
    x: torch.Tensor,
    divisor: int = 32,
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad tensor to be divisible by divisor"""
    if x.dim() == 3:
        C, H, W = x.shape
    else:
        H, W = x.shape

    padH = (divisor - H % divisor) % divisor
    padW = (divisor - W % divisor) % divisor

    if padH == 0 and padW == 0:
        return x, (0, 0, 0, 0)

    x = F.pad(x, (0, padW, 0, padH), value=pad_value)
    return x, (0, padW, 0, padH)


def _is_closed_polygon(pts_xy: np.ndarray, close_tol: float = 5.0, min_area: float = 50.0) -> bool:
    """Check if points form a closed polygon"""
    if pts_xy.shape[0] < 3:
        return False
    p0, pN = pts_xy[0], pts_xy[-1]
    if np.linalg.norm(p0 - pN) > close_tol:
        return False
    area = cv2.contourArea(pts_xy.astype(np.float32))
    return abs(area) >= min_area


def make_lane_mask(
    json_path: str,
    H: int, W: int,
    line_width: int = 6,
    use_aa: bool = False
) -> np.ndarray:
    """Create binary lane mask from JSON polygon/polyline annotations"""
    with open(json_path, "r") as f:
        ann = json.load(f)

    geoms = ann.get("geometry", [])
    mask = np.zeros((H, W), dtype=np.uint8)
    lt = cv2.LINE_AA if use_aa else cv2.LINE_8

    for g in geoms:
        if not g or len(g) < 2:
            continue
        pts = np.array(g, dtype=np.float32)
        pts_i = np.round(pts).astype(np.int32)

        if _is_closed_polygon(pts, close_tol=5.0, min_area=50.0):
            cv2.fillPoly(mask, [pts_i], 1)
        else:
            cv2.polylines(mask, [pts_i], isClosed=False, color=1, thickness=line_width, lineType=lt)

    return mask


def build_lane_items(
    split_root: str,
    exclude_folder: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Build list of (image_path, json_path) tuples"""
    img_root = os.path.join(split_root, "images")
    lbl_root = os.path.join(split_root, "labels")

    folders = sorted([
        d for d in os.listdir(img_root)
        if os.path.isdir(os.path.join(img_root, d))
    ])

    items = []
    for folder in folders:
        if exclude_folder is not None and folder == exclude_folder:
            continue

        img_dir = os.path.join(img_root, folder)
        lbl_dir = os.path.join(lbl_root, folder)

        if not os.path.isdir(lbl_dir):
            continue

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        img_paths += sorted(glob.glob(os.path.join(img_dir, "*.png")))

        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(lbl_dir, f"{stem}.json")
            if os.path.isfile(json_path):
                items.append((img_path, json_path))

    return items


class SDLaneDataset(Dataset):
    """SDLane Dataset for binary lane segmentation"""

    def __init__(
        self,
        items: List[Tuple[str, str]],
        shortest_edge: int = 1024,
        longest_edge: int = 1920,
        size_divisor: int = 32,
        line_width: int = 6,
        use_aa: bool = False,
        return_meta: bool = False,
        use_crop: bool = False,
        crop_size: int = 1024
    ):
        self.items = items
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.size_divisor = size_divisor
        self.line_width = line_width
        self.use_aa = use_aa
        self.return_meta = return_meta
        self.use_crop = use_crop
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.items)

    def _random_crop(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        crop_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random crop with padding if needed"""
        h, w = mask.shape
        if h < crop_size or w < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            img = cv2.copyMakeBorder(
                img, 0, pad_h, 0, pad_w,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            h, w = mask.shape

        y0 = random.randint(0, h - crop_size)
        x0 = random.randint(0, w - crop_size)
        return img[y0:y0+crop_size, x0:x0+crop_size], mask[y0:y0+crop_size, x0:x0+crop_size]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, json_path = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        gt_mask = make_lane_mask(json_path, H, W, self.line_width, self.use_aa)

        newH, newW = resize_keep_aspect(H, W, self.shortest_edge, self.longest_edge)
        img_np = np.array(img)
        img_rs = cv2.resize(img_np, (newW, newH), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(gt_mask, (newW, newH), interpolation=cv2.INTER_NEAREST)

        if (not self.return_meta) and self.use_crop:
            img_rs, mask_rs = self._random_crop(img_rs, mask_rs, self.crop_size)

        x = torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = torch.from_numpy(mask_rs).long()

        x, pad = pad_to_divisor(x, self.size_divisor, 0.0)
        y, _ = pad_to_divisor(y.unsqueeze(0).float(), self.size_divisor, 0.0)
        y = y.squeeze(0).long()

        out = {"pixel_values": x, "labels": y}

        if self.return_meta:
            out["orig_hw"] = (H, W)
            out["resized_hw"] = (newH, newW)
            out["pad"] = pad
            out["gt_orig"] = torch.from_numpy(gt_mask).to(torch.uint8)

        return out


def train_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for training"""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }


def val_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for validation (batch_size=1)"""
    b = batch[0]
    return {
        "pixel_values": b["pixel_values"].unsqueeze(0),
        "labels": b["labels"].unsqueeze(0),
        "orig_hw": b["orig_hw"],
        "resized_hw": b["resized_hw"],
        "pad": b["pad"],
        "gt_orig": b["gt_orig"],
    }
