import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal


def compute_class_weights(
    class_info_path: str,
    num_classes: int = 62,
    method: Literal["inverse", "effective", "sqrt_inverse"] = "inverse",
    beta: float = 0.999,
    normalize: bool = True
) -> torch.Tensor:
    """
    클래스 불균형을 위한 class weights 계산

    Args:
        class_info_path: class_info.csv 파일 경로 (class_id, class_name, count)
        num_classes: 전체 클래스 수
        method: 계산 방식
            - "inverse": weight = total / (num_classes * count)
            - "effective": weight = (1 - beta) / (1 - beta^count) (논문 기반)
            - "sqrt_inverse": weight = sqrt(total / count) (완화된 버전)
        beta: effective number 방식에서 사용 (default: 0.999)
        normalize: weights 정규화 여부 (합이 num_classes가 되도록)

    Returns:
        class_weights: [num_classes] 크기의 weight tensor
    """
    # CSV 로드
    df = pd.read_csv(class_info_path)

    # class_id -> count 매핑
    id_to_count = dict(zip(df['class_id'], df['count']))

    # 모든 클래스에 대해 count 가져오기 (없으면 1로 설정)
    counts = np.array([id_to_count.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    total = counts.sum()

    # 방식별 weight 계산
    if method == "inverse":
        # Inverse Frequency
        weights = total / (num_classes * counts)

    elif method == "effective":
        # Effective Number of Samples (CVPR 2019)
        # "Class-Balanced Loss Based on Effective Number of Samples"
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num

    elif method == "sqrt_inverse":
        # Square root inverse (완화된 버전)
        weights = np.sqrt(total / counts)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse', 'effective', or 'sqrt_inverse'")

    # 정규화 (선택)
    if normalize:
        weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def get_class_weights(
    config: dict,
    num_classes: int,
    data_root: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    Config에서 class weights 설정을 읽어 weights 반환

    Args:
        config: 전체 config dict
        num_classes: 클래스 수
        data_root: 데이터 루트 경로 (class_info.csv 위치)

    Returns:
        class_weights tensor 또는 None (비활성화시)
    """
    loss_cfg = config.get("loss", {})
    cw_cfg = loss_cfg.get("class_weights", {})

    # 비활성화시 None 반환
    if not cw_cfg.get("enabled", False):
        return None

    # class_info.csv 경로 결정
    if data_root:
        # data_root 상위에서 찾기
        possible_paths = [
            Path(data_root).parent / "multi_class_info.csv",
            Path(data_root) / "class_info.csv",
            Path("dataset") / "multi_class_info.csv",
        ]
    else:
        possible_paths = [
            Path("dataset") / "multi_class_info.csv",
            Path("dataset/Multi_Camera_Semantic_Segmentation") / "class_info.csv",
        ]

    class_info_path = None
    for p in possible_paths:
        if p.exists():
            class_info_path = str(p)
            break

    if class_info_path is None:
        raise FileNotFoundError(f"class_info.csv not found. Searched: {possible_paths}")

    # weights 계산
    weights = compute_class_weights(
        class_info_path=class_info_path,
        num_classes=num_classes,
        method=cw_cfg.get("method", "inverse"),
        beta=cw_cfg.get("beta", 0.999),
        normalize=cw_cfg.get("normalize", True)
    )

    return weights
