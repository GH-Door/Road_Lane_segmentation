import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import Inferencer, visualize_prediction


def demo_single_image(image_path: str, checkpoint: str = "checkpoints/best.pt"):
    """단일 이미지 추론 데모"""
    inferencer = Inferencer(checkpoint=checkpoint)

    # 이미지 로드
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 추론
    mask = inferencer.predict(image)

    # 시각화
    visualize_prediction(image, mask, num_classes=inferencer.num_classes)

    return mask


def demo_from_dataset(data_root: str, split: str = "val", idx: int = 0, checkpoint: str = "checkpoints/best.pt"):
    """데이터셋에서 샘플 추론 데모"""
    from src.data import DatasetLoader
    from src.inference import visualize_comparison

    inferencer = Inferencer(checkpoint=checkpoint)

    # 데이터셋 로드 (transform 없이 원본)
    dataset = DatasetLoader(data_root=data_root, split=split, transform=None)

    # 샘플 가져오기
    img_tensor, gt_mask = dataset[idx]
    image = img_tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    image = (image * 255).astype(np.uint8)
    gt_mask = gt_mask.numpy()

    # 추론
    pred_mask = inferencer.predict(image)

    # 비교 시각화
    visualize_comparison(image, pred_mask, gt_mask, num_classes=inferencer.num_classes)

    return pred_mask, gt_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="이미지 경로")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data_root", type=str, default="dataset/Multi Camera Semantic Segmentation")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    if args.image:
        demo_single_image(args.image, args.checkpoint)
    else:
        demo_from_dataset(args.data_root, args.split, args.idx, args.checkpoint)
