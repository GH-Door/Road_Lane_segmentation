import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet pretrained 표준값 (DeepLabV3+ ResNet backbone용)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def get_transforms(img_size: tuple = (1024, 768)):
    preprocessing = [
        A.Resize(
            height=img_size[0], 
            width=img_size[1], 
            interpolation=cv2.INTER_LINEAR,       # 이미지용
            mask_interpolation=cv2.INTER_NEAREST # 마스크용 (값 보존) mask_interpolation을 NEAREST로 설정하는 것이 핵심
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ]
    return A.Compose(preprocessing)