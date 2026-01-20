import json
import glob
import numpy as np
import cv2
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from torch.utils.data import Dataset


class TestDatasetLoader(Dataset):
    """Mono Camera Test Dataset (평면 구조)"""

    def __init__(
        self,
        data_root: str,
        transform=None,
    ):
        """
        Args:
            data_root: 테스트 데이터셋 루트 경로
            transform: albumentations transform
        """
        self.data_root = Path(data_root)
        self.transform = transform

        self.img_dir = self.data_root / "JPEGImages_mosaic"
        self.labels_dir = self.data_root / "labels"

        # 클래스 매핑 로드
        self.classes = self.load_classes()
        self.num_classes = len(self.classes)

        # 파일 목록 로드
        self.samples = self.load_samples()

    def load_classes(self) -> Dict[str, int]:
        """클래스 매핑 로드 (data_root/class_info.csv)"""
        csv_path = self.data_root / "class_info.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"class_info.csv not found: {csv_path}")
        class_df = pd.read_csv(csv_path)
        return dict(zip(class_df['class_name'], class_df['class_id']))

    def load_samples(self) -> List[Dict]:
        """이미지-라벨 쌍 로드 (평면 구조)"""
        samples = []
        label_files = sorted(glob.glob(str(self.labels_dir / "*.json")))

        for label_path in label_files:
            label_path = Path(label_path)

            # 이미지 경로 생성
            filename = label_path.stem.replace("_gtFine_polygons", "")
            img_path = self.img_dir / f"{filename}.png"

            if img_path.exists():
                samples.append({
                    "img_path": str(img_path),
                    "label_path": str(label_path),
                    "filename": filename,
                })

        print(f"Test samples: {len(samples)}")
        return samples

    def load_json(self, json_path: str) -> dict:
        """JSON 어노테이션 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def polygon_to_mask(self, annotation: dict, img_size: Tuple[int, int]) -> np.ndarray:
        """폴리곤 어노테이션 → 세그멘테이션 마스크"""
        h, w = img_size
        mask = np.zeros((h, w), dtype=np.uint8)

        for obj in annotation.get("objects", []):
            if obj.get("deleted", 0) == 1:
                continue

            label = obj.get("label", "unlabeled")
            class_id = self.classes.get(label, 0)

            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue

            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], class_id)

        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # 이미지 로드 (한글 경로 지원)
        img = cv2.imdecode(np.fromfile(sample["img_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {sample['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 라벨 로드 및 마스크 생성
        annotation = self.load_json(sample["label_path"])
        orig_h, orig_w = annotation["imgHeight"], annotation["imgWidth"]
        mask = self.polygon_to_mask(annotation, (orig_h, orig_w))

        # Transform 적용
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].long()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return img, mask

    def get_sample_info(self, idx: int) -> Dict:
        """샘플 정보 반환"""
        return self.samples[idx]
