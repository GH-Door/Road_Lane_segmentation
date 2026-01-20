import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def create_colormap(num_classes: int = 62) -> np.ndarray:
    """클래스별 컬러맵 생성"""
    np.random.seed(42)
    colormap = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # 배경은 검정
    return colormap


def mask_to_rgb(mask: np.ndarray, num_classes: int = 62) -> np.ndarray:
    """
    마스크를 RGB 이미지로 변환

    Args:
        mask: (H, W) 클래스 인덱스
        num_classes: 클래스 수

    Returns:
        rgb: (H, W, 3) RGB 이미지
    """
    colormap = create_colormap(num_classes)
    rgb = colormap[mask]
    return rgb


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    num_classes: int = 62
) -> np.ndarray:
    """
    이미지에 마스크 오버레이

    Args:
        image: (H, W, 3) RGB 이미지
        mask: (H, W) 클래스 인덱스
        alpha: 투명도 (0~1)
        num_classes: 클래스 수

    Returns:
        overlay: (H, W, 3) 오버레이된 이미지
    """
    mask_rgb = mask_to_rgb(mask, num_classes)
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return overlay


def visualize_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    num_classes: int = 62,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    예측 결과 시각화

    Args:
        image: (H, W, 3) RGB 이미지
        mask: (H, W) 예측 마스크
        num_classes: 클래스 수
        figsize: figure 크기
        save_path: 저장 경로 (None이면 저장 안 함)
    """
    mask_rgb = mask_to_rgb(mask, num_classes)
    overlay = overlay_mask(image, mask, alpha=0.5, num_classes=num_classes)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def visualize_comparison(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = 62,
    figsize: Tuple[int, int] = (20, 5),
    save_path: Optional[str] = None
):
    """
    예측 vs Ground Truth 비교 시각화

    Args:
        image: (H, W, 3) RGB 이미지
        pred_mask: (H, W) 예측 마스크
        gt_mask: (H, W) 정답 마스크
        num_classes: 클래스 수
        figsize: figure 크기
        save_path: 저장 경로
    """
    pred_rgb = mask_to_rgb(pred_mask, num_classes)
    gt_rgb = mask_to_rgb(gt_mask, num_classes)
    overlay = overlay_mask(image, pred_mask, alpha=0.5, num_classes=num_classes)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
