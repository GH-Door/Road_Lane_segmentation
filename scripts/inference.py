import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import Inferencer, visualize_prediction
from src.inference.visualize import mask_to_rgb # mask_to_rgb 추가


def single_image_inference(image_path: str, checkpoint: str = "checkpoints/best.pt", save_dir: Optional[str] = None, save_idx: int = 0):
    """단일 이미지 추론"""
    inferencer = Inferencer(checkpoint=checkpoint)

    # 이미지 로드
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 추론
    mask = inferencer.predict(image)

    # 마스크 저장
    if save_dir:
        full_save_dir = os.path.join("inference", save_dir) # "inference/" 접두어 추가
        os.makedirs(full_save_dir, exist_ok=True)
        filename = Path(image_path).stem + f"_pred_mask_{save_idx}.png"
        save_path = os.path.join(full_save_dir, filename)
        mask_rgb = mask_to_rgb(mask, num_classes=inferencer.num_classes)
        cv2.imwrite(save_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved predicted mask to {save_path}")

    # 시각화
    visualize_prediction(image, mask, num_classes=inferencer.num_classes)
    return mask


def dataset_inference(data_root: str, split: str = "val", idx: int = 0, checkpoint: str = "checkpoints/best.pt", save_dir: Optional[str] = None, save_idx: int = 0):
    """데이터셋에서 샘플 추론"""
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

    # 마스크 저장
    if save_dir:
        full_save_dir = os.path.join("inference", save_dir) # "inference/" 접두어 추가
        os.makedirs(full_save_dir, exist_ok=True)
        filename = f"dataset_{split}_idx_{idx}_pred_mask_{save_idx}.png"
        save_path = os.path.join(full_save_dir, filename)
        mask_rgb = mask_to_rgb(pred_mask, num_classes=inferencer.num_classes)
        cv2.imwrite(save_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved predicted mask to {save_path}")

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
    parser.add_argument("--save-dir", type=str, help="예측 마스크를 저장할 디렉토리 경로", default=None) # save-dir 인자 추가
    parser.add_argument("--num-to-save", type=int, default=1, help="저장할 이미지 수 (데모 이미지 수와 동일)") # num-to-save 인자 추가
    args = parser.parse_args()

    if args.image:
        single_image_inference(args.image, args.checkpoint, args.save_dir, 0) # save_idx는 0으로 고정
    else:
        # 데이터셋에서 여러 이미지 저장
        if args.save_dir and args.num_to_save > 1:
            for i in range(args.num_to_save):
                print(f"Processing dataset image {i+1}/{args.num_to_save}")
                dataset_inference(args.data_root, args.split, i, args.checkpoint, args.save_dir, i)
        else:
            dataset_inference(args.data_root, args.split, args.idx, args.checkpoint, args.save_dir, args.idx)
