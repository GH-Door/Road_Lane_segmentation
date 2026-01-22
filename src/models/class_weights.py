import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal, Dict


def compute_class_weights(
    class_info_path: str,
    num_classes: int = 62,
    method: Literal["inverse", "effective", "sqrt_inverse"] = "inverse",
    beta: float = 0.999,
    normalize: bool = True,
    group_mapping: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    클래스 불균형을 위한 class weights 계산

    Args:
        class_info_path: class_info.csv 파일 경로 (class_name, count)
        num_classes: 전체 클래스 수 (그룹화 시 그룹 수)
        method: 계산 방식
            - "inverse": weight = total / (num_classes * count)
            - "effective": weight = (1 - beta) / (1 - beta^count) (논문 기반)
            - "sqrt_inverse": weight = sqrt(total / count) (완화된 버전)
        beta: effective number 방식에서 사용 (default: 0.999)
        normalize: weights 정규화 여부 (합이 num_classes가 되도록)
        group_mapping: 그룹화 매핑 {class_name: group_id} (None이면 원본 class_id 사용)

    Returns:
        class_weights: [num_classes] 크기의 weight tensor
    """
    # CSV 로드
    df = pd.read_csv(class_info_path)

    if group_mapping:
        # 그룹화: class_name → group_id로 count 합산
        group_counts = np.zeros(num_classes, dtype=np.float32)
        for _, row in df.iterrows():
            class_name = row['class_name']
            count = row['count']
            group_id = group_mapping.get(class_name, 0)  # 매핑 없으면 0 (무시 그룹)
            if 0 <= group_id < num_classes:
                group_counts[group_id] += count
        counts = group_counts
        # 0인 그룹은 1로 설정 (division by zero 방지)
        counts = np.where(counts == 0, 1, counts)
    else:
        # 원본: class_id 기준
        if 'class_id' in df.columns:
            id_to_count = dict(zip(df['class_id'], df['count']))
        else:
            id_to_count = {idx: row['count'] for idx, row in df.iterrows()}
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


def apply_manual_boost(
    weights: torch.Tensor,
    multipliers: Dict[int, float],
    normalize: bool = True
) -> torch.Tensor:
    """
    수동 가중치 보정 적용 (Confusion Matrix 기반)

    Args:
        weights: 기본 가중치 tensor [num_classes]
        multipliers: 클래스별 boost 배율 {class_id: multiplier}
            - multiplier > 1.0: 해당 클래스 중요도 증가 (오분류가 많은 클래스)
            - multiplier < 1.0: 해당 클래스 중요도 감소 (데이터가 많고 잘 분류되는 클래스)
        normalize: 최종 정규화 여부 (전체 합 = num_classes)

    Returns:
        boosted_weights: boost 적용된 가중치 tensor

    Example:
        # Confusion Matrix에서 유도선(8)이 54% 일반도로로 오분류 → boost 10.0
        # 무시(0)는 데이터가 많고 99% 정확 → boost 0.2
        multipliers = {
            0: 0.2,   # 무시 (감소)
            1: 0.5,   # 일반도로 (감소)
            3: 5.0,   # 중앙선 (증가)
            8: 10.0,  # 유도선 (최대 증가)
        }
    """
    weights = weights.clone()
    num_classes = len(weights)

    # 1. boost 배율 적용
    for cls_id, multiplier in multipliers.items():
        if 0 <= cls_id < num_classes:
            weights[cls_id] *= multiplier

    # 2. 최종 정규화 (전체 합 = num_classes)
    if normalize:
        weights = weights / weights.sum() * num_classes

    return weights


def get_class_weights(
    config: dict,
    num_classes: int,
    data_root: Optional[str] = None,
    group_config: Optional[dict] = None
) -> Optional[torch.Tensor]:
    """
    Config에서 class weights 설정을 읽어 weights 반환

    Args:
        config: 전체 config dict
        num_classes: 클래스 수 (그룹화 시 그룹 수)
        data_root: 데이터 루트 경로
        group_config: 그룹화 설정 dict (enabled, mapping 포함)

    Returns:
        class_weights tensor 또는 None (비활성화시)
    """
    loss_cfg = config.get("loss", {})
    cw_cfg = loss_cfg.get("class_weights", {})

    # 비활성화시 None 반환
    if not cw_cfg.get("enabled", False):
        return None

    # class_info 경로 (config에서 먼저 확인)
    data_cfg = config.get("data", {})
    class_info_path = data_cfg.get("class_info")

    if not class_info_path or not Path(class_info_path).exists():
        # fallback 경로
        if data_root:
            possible_paths = [
                Path(data_root).parent / "total_class_info.csv",
                Path(data_root) / "class_info.csv",
            ]
        else:
            possible_paths = [
                Path("dataset") / "total_class_info.csv",
            ]

        for p in possible_paths:
            if p.exists():
                class_info_path = str(p)
                break

    if class_info_path is None or not Path(class_info_path).exists():
        raise FileNotFoundError(f"class_info.csv not found")

    # 그룹화 매핑 추출
    group_mapping = None
    if group_config and group_config.get("enabled", False):
        group_mapping = group_config.get("mapping")

    # weights 계산
    weights = compute_class_weights(
        class_info_path=class_info_path,
        num_classes=num_classes,
        method=cw_cfg.get("method", "inverse"),
        beta=cw_cfg.get("beta", 0.999),
        normalize=cw_cfg.get("normalize", True),
        group_mapping=group_mapping
    )

    # 수동 boost 적용 (선택)
    manual_boost_cfg = cw_cfg.get("manual_boost", {})
    if manual_boost_cfg.get("enabled", False):
        multipliers = manual_boost_cfg.get("multipliers", {})
        if multipliers:
            # YAML에서 키가 문자열로 들어올 수 있으므로 int로 변환
            multipliers = {int(k): float(v) for k, v in multipliers.items()}
            weights = apply_manual_boost(
                weights=weights,
                multipliers=multipliers,
                normalize=cw_cfg.get("normalize", True)
            )
            print(f"[ClassWeights] Manual boost applied: {multipliers}")

    return weights
