import albumentations as A
import warnings
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings("ignore")

# ImageNet pretrained 표준값
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_augmentation(cfg, img_size):
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", False):
        return A.Compose([
            A.Resize(*img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])

    transforms = []

    geo = aug_cfg.get("geometric", {})
    if geo.get("enabled", False):
        if geo.get("horizontal_flip", False):
            transforms.append(A.HorizontalFlip(p=0.15))

        if geo.get("rotate", {}).get("enabled", False):
            transforms.append(
                A.Rotate(
                    limit=geo["rotate"].get("limit", 3),
                    border_mode=0,
                    p=0.15
                )
            )

        if geo.get("scale", {}).get("enabled", False):
            transforms.append(
                A.RandomScale(
                    scale_limit=geo["scale"].get("scale_limit", 0.03),
                    p=0.1
                )
            )

    photo = aug_cfg.get("photometric", {})
    if photo.get("enabled", False):
        if photo.get("brightness_contrast", {}).get("enabled", False):
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=photo["brightness_contrast"].get("brightness_limit", 0.15),
                    contrast_limit=photo["brightness_contrast"].get("contrast_limit", 0.15),
                    p=0.3
                )
            )

        if photo.get("hue_saturation", {}).get("enabled", False):
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=photo["hue_saturation"].get("hue_shift_limit", 5),
                    sat_shift_limit=photo["hue_saturation"].get("sat_shift_limit", 5),
                    val_shift_limit=photo["hue_saturation"].get("val_shift_limit", 5),
                    p=0.15
                )
            )

    night = aug_cfg.get("night_simulation", {})
    if night.get("enabled", False):
        if night.get("gamma", {}).get("enabled", False):
            transforms.append(
                A.RandomGamma(
                    gamma_limit=night["gamma"].get("gamma_limit", [85, 115]),
                    p=0.15
                )
            )

        if night.get("shadow", {}).get("enabled", False):
            transforms.append(
                A.RandomShadow(p=0.08)
            )

    nb = aug_cfg.get("noise_blur", {})
    if nb.get("enabled", False):
        if nb.get("gaussian_noise", {}).get("enabled", False):
            transforms.append(
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.08)
            )

        if nb.get("motion_blur", {}).get("enabled", False):
            transforms.append(
                A.MotionBlur(
                    blur_limit=3,
                    p=0.08
                )
            )

    transforms.extend([
        A.Resize(*img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    return A.Compose(transforms)