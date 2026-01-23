import json
import glob
import numpy as np
import cv2
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    """Multi-Camera Semantic Segmentation Dataset"""
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        camera: str = "left",
        transform=None,
        class_info_path: Optional[str] = None,
        class_grouping: Optional[Dict[str, int]] = None,
        num_grouped_classes: Optional[int] = None,
        train_batch_size: int = 1,
    ):
        """
        Args:
            data_root: 데이터셋 루트 경로
            split: 'train' or 'val'
            camera: 'left' or 'right'
            transform: albumentations transform
            class_info_path: 클래스 정보 CSV 파일 경로 (지정하지 않으면 data_root에서 찾음)
            class_grouping: 클래스명 → 그룹ID 매핑 (그룹화 비활성화 시 None)
            num_grouped_classes: 그룹화 후 클래스 수 (그룹화 활성화 시 필수)
            train_batch_size: 학습 배치 크기 (첫 배치 로딩 완료 로그용)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.camera = camera
        self.transform = transform
        self.class_info_path = class_info_path
        self.class_grouping = class_grouping  # 클래스명 → 그룹ID
        self.num_grouped_classes = num_grouped_classes
        self.train_batch_size = train_batch_size


        # 경로 설정
        self.labels_dir = self.data_root / "labels" / split
        self.img_dir = self.data_root / f"{camera}Img" / split

        # 클래스 매핑 로드 (원본 클래스명 → 원본 ID)
        self.classes = self.load_classes()

        # 그룹화 활성화 시 num_classes는 그룹 수, 아니면 원본 클래스 수
        if self.class_grouping and self.num_grouped_classes:
            self.num_classes = self.num_grouped_classes
        else:
            self.num_classes = len(self.classes)

        # 파일 목록 로드
        self.samples = self.load_samples()

    def load_classes(self) -> Dict[str, int]:
        """클래스 라벨 → ID 매핑 (CSV에서 로드)"""
        if self.class_info_path:
            csv_path = Path(self.class_info_path)
        else:
            csv_path = self.data_root / "class_info.csv"

        if csv_path.exists():
            class_df = pd.read_csv(csv_path)
            # class_id 컬럼이 없으면 인덱스를 class_id로 사용
            if 'class_id' in class_df.columns:
                classes = dict(zip(class_df['class_name'], class_df['class_id']))
            else:
                classes = {name: idx for idx, name in enumerate(class_df['class_name'])}
        else:
            # CSV 없으면 생성
            print(f"{csv_path} not found. Generating from {self.data_root}...")
            class_df = generate_class_info(str(self.data_root), save=True)
            classes = dict(zip(class_df['class_name'], class_df['class_id']))

        return classes

    def load_samples(self) -> List[Dict]:
        """이미지-라벨 쌍 로드"""
        samples = []
        print(f"INFO: Scanning for JSON files in {self.labels_dir}...")
        label_files = sorted(glob.glob(str(self.labels_dir / "**/*.json"), recursive=True))

        for label_path in label_files:
            label_path = Path(label_path)

            # 이미지 경로 생성
            rel_path = label_path.relative_to(self.labels_dir)
            folder = rel_path.parent
            filename = rel_path.stem.replace("_gtFine_polygons", "")

            img_suffix = "leftImg8bit" if self.camera == "left" else "rightImg8bit"
            img_path = self.img_dir / folder / f"{filename}_{img_suffix}.png"

            if img_path.exists():
                samples.append({
                    "img_path": str(img_path),
                    "label_path": str(label_path),
                    "folder": str(folder),
                })

        print(f"{self.split.capitalize()} samples: {len(samples)}")

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

            # 그룹화 적용: class_grouping이 있으면 그룹ID, 없으면 원본 class_id
            if self.class_grouping:
                class_id = self.class_grouping.get(label, 0)  # 매핑 없으면 0 (무시 그룹)
            else:
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
        # if idx == 0:
        #     # Print only once for the first item requested by any worker.
        #     print(f"INFO: DatasetLoader: Starting data processing for split '{self.split}' (first index: {idx}).")

        try:
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

            # if idx == self.train_batch_size - 1:
            #     # Print only once after the last item of the first batch is processed.
            #     print(f"INFO: DatasetLoader: First batch prepared for split '{self.split}'. Tqdm progress bar will appear shortly.")

            return img, mask

        except Exception as e:
            # Raising the exception is important for the DataLoader to handle it.
            print(f"ERROR: DatasetLoader: Failed to load item at Index: {idx} | File: {self.samples[idx]['img_path']} | Error: {str(e)}")
            raise e

    def get_sample_info(self, idx: int) -> Dict:
        """샘플 정보 반환 (디버깅/시각화용)"""
        return self.samples[idx]


def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 4,
    transform=None,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> torch.utils.data.DataLoader:
    """DataLoader 생성 헬퍼 함수"""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = DatasetLoader(
        data_root=data_root,
        split=split,
        transform=transform,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def generate_class_info(data_root: str, save: bool = True) -> pd.DataFrame:
    """JSON 파일들에서 클래스 정보 추출 후 CSV 저장"""
    from collections import Counter

    data_root = Path(data_root)
    class_counts = Counter()

    # train, val 모두 확인
    for split in ['train', 'val']:
        labels_dir = data_root / "labels" / split
        label_files = glob.glob(str(labels_dir / "**/*.json"), recursive=True)

        for label_path in tqdm(label_files, desc=f"Scanning {split}"):
            with open(label_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            for obj in annotation.get("objects", []):
                if obj.get("deleted", 0) == 0:
                    label = obj.get("label", "unlabeled")
                    class_counts[label] += 1

    # DataFrame 생성 (등장 횟수 기준 정렬)
    class_df = pd.DataFrame(
        class_counts.most_common(),
        columns=['class_name', 'count']
    )
    class_df['class_id'] = range(len(class_df))
    class_df = class_df[['class_id', 'class_name', 'count']]

    # CSV 저장
    if save:
        csv_path = data_root / "class_info.csv"
        class_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    return class_df


def get_dataset_info(data_root: str, split: str = "train") -> pd.DataFrame:
    data_root = Path(data_root)
    labels_dir = data_root / "labels" / split
    img_dir = data_root / "leftImg" / split
    label_files = sorted(glob.glob(str(labels_dir / "**/*.json"), recursive=True))

    records = []
    for label_path in tqdm(label_files, desc=f"Loading {split}"):
        label_path = Path(label_path)

        # 이미지 경로 생성
        rel_path = label_path.relative_to(labels_dir)
        folder = rel_path.parent
        filename = rel_path.stem.replace("_gtFine_polygons", "")
        img_path = img_dir / folder / f"{filename}_leftImg8bit.png"

        # JSON 로드
        with open(label_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # 클래스 추출 (deleted=0인 것만)
        classes = []
        num_objects = 0
        for obj in annotation.get("objects", []):
            if obj.get("deleted", 0) == 0:
                classes.append(obj.get("label", "unlabeled"))
                num_objects += 1

        records.append({
            "img_path": str(img_path),
            "label_path": str(label_path),
            "folder": str(folder),
            "split": split,
            "width": annotation.get("imgWidth", 0),
            "height": annotation.get("imgHeight", 0),
            "classes": classes,
            "num_objects": num_objects,
            "unique_classes": list(set(classes)),
            "num_classes": len(set(classes)),
        })

    return pd.DataFrame(records)


def split_dataset(dataset: DatasetLoader, test_ratio: float = 0.2, seed: int = 42):
    """
    Dataset을 train/test 또는 val/test로 분할

    Args:
        dataset: DatasetLoader 인스턴스
        test_ratio: test 비율 (0.2 = 20%)
        seed: random seed

    Returns:
        (train_dataset, test_dataset) 튜플
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed
    )

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    return train_subset, test_subset
