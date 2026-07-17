import numpy as np
from PIL import Image
import cv2
import random
import gc
from typing import List, Tuple, Dict, Any

def apply_camera_grain(pil_img: Image.Image) -> Image.Image:
    img_np = np.array(pil_img)
    h, w, c = img_np.shape
    noise = np.random.normal(0, 8, (h, w, c))
    noisy = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_salt_pepper_noise(pil_img: Image.Image, amount: float = 0.01) -> Image.Image:
    img_np = np.array(pil_img)
    h, w, c = img_np.shape
    noisy = img_np.copy()
    n = int(amount * h * w)
    
    noisy[np.random.randint(0, h, n), np.random.randint(0, w, n)] = 255  # salt
    noisy[np.random.randint(0, h, n), np.random.randint(0, w, n)] = 0    # pepper
    return Image.fromarray(noisy)

def apply_gaussian_noise(pil_img: Image.Image, sigma: float = 15.0) -> Image.Image:
    img_np = np.array(pil_img)
    noise = np.random.normal(0, sigma, img_np.shape)
    noisy = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_blur(pil_img: Image.Image, ksize: int = 5) -> Image.Image:
    img_np = np.array(pil_img)
    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
    return Image.fromarray(blurred)

def flip_obb_labels_horizontal(coords: List[float]) -> List[float]:
    return [
        1.0 - coords[2], coords[3],
        1.0 - coords[0], coords[1],
        1.0 - coords[6], coords[7],
        1.0 - coords[4], coords[5]
    ]

def flip_obb_labels_vertical(coords: List[float]) -> List[float]:
    return [
        coords[6], 1.0 - coords[7],
        coords[4], 1.0 - coords[5],
        coords[2], 1.0 - coords[3],
        coords[0], 1.0 - coords[1]
    ]

def flip_obb_labels_both(coords: List[float]) -> List[float]:
    return [
        1.0 - coords[4], 1.0 - coords[5],
        1.0 - coords[6], 1.0 - coords[7],
        1.0 - coords[0], 1.0 - coords[1],
        1.0 - coords[2], 1.0 - coords[3]
    ]

def flip_yolo_labels_horizontal(coords: List[float]) -> List[float]:
    return [1.0 - coords[0], coords[1], coords[2], coords[3]]

def flip_yolo_labels_vertical(coords: List[float]) -> List[float]:
    return [coords[0], 1.0 - coords[1], coords[2], coords[3]]

def flip_yolo_labels_both(coords: List[float]) -> List[float]:
    return [1.0 - coords[0], 1.0 - coords[1], coords[2], coords[3]]

def _flip_coords(coords: List[float], direction: str, project_type: str) -> List[float]:
    if project_type == "Yolo OBB":
        fn = {"h": flip_obb_labels_horizontal,
              "v": flip_obb_labels_vertical,
              "hv": flip_obb_labels_both}[direction]
    else:
        fn = {"h": flip_yolo_labels_horizontal,
              "v": flip_yolo_labels_vertical,
              "hv": flip_yolo_labels_both}[direction]
    return fn(coords)

def augment_image_and_labels(
    pil_img: Image.Image,
    labels: List[Tuple[int, List[float]]],
    project_type: str,
    opts: Dict[str, Any],
) -> List[Tuple[Image.Image, List[Tuple[int, List[float]]], str]]:
    
    results = []
    enabled = []
    if getattr(opts, "flip_h", False):  enabled.append("flip_h")
    if getattr(opts, "flip_v", False):  enabled.append("flip_v")
    if getattr(opts, "flip_hv", False): enabled.append("flip_hv")
    if getattr(opts, "grain", False):   enabled.append("grain")
    if getattr(opts, "noise", False):   enabled.append("noise")
    if getattr(opts, "blur", False):    enabled.append("blur")

    if not enabled:
        return results

    num_augs = getattr(opts, "num_augs", 3)
    
    for k in range(num_augs):
        aug_img = pil_img.copy()
        aug_labels = list(labels)
        applied = []

        if enabled:
            n = random.randint(1, min(3, len(enabled)))
            chosen = random.sample(enabled, n)

            flips = [t for t in chosen if t.startswith("flip")]
            if flips:
                flip = random.choice(flips)
                if flip == "flip_h":
                    aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                    aug_labels = [(c, _flip_coords(p, "h", project_type)) for c, p in aug_labels]
                    applied.append("FlipH")
                elif flip == "flip_v":
                    aug_img = aug_img.transpose(Image.FLIP_TOP_BOTTOM)
                    aug_labels = [(c, _flip_coords(p, "v", project_type)) for c, p in aug_labels]
                    applied.append("FlipV")
                elif flip == "flip_hv":
                    aug_img = aug_img.transpose(Image.ROTATE_180)
                    aug_labels = [(c, _flip_coords(p, "hv", project_type)) for c, p in aug_labels]
                    applied.append("FlipHV")

            if "grain" in chosen:
                aug_img = apply_camera_grain(aug_img)
                applied.append("Grain")
            if "noise" in chosen:
                if random.random() < 0.5:
                    aug_img = apply_salt_pepper_noise(aug_img, amount=random.uniform(0.008, 0.02))
                    applied.append("SaltPepper")
                else:
                    aug_img = apply_gaussian_noise(aug_img, sigma=random.uniform(10, 20))
                    applied.append("GaussNoise")
            if "blur" in chosen:
                ksize = random.choice([3, 5])
                aug_img = apply_blur(aug_img, ksize)
                applied.append(f"BlurK{ksize}")
        
        results.append((aug_img, aug_labels, f"_aug_{k+1}"))
        gc.collect()
        
    return results

def augment_image_only(
    pil_img: Image.Image,
    opts: Dict[str, Any],
) -> List[Tuple[Image.Image, str]]:
    
    results = []
    enabled = []
    if getattr(opts, "flip_h", False):  enabled.append("flip_h")
    if getattr(opts, "flip_v", False):  enabled.append("flip_v")
    if getattr(opts, "flip_hv", False): enabled.append("flip_hv")
    if getattr(opts, "grain", False):   enabled.append("grain")
    if getattr(opts, "noise", False):   enabled.append("noise")
    if getattr(opts, "blur", False):    enabled.append("blur")

    if not enabled:
        return results

    num_augs = getattr(opts, "num_augs", 3)
    
    for k in range(num_augs):
        aug_img = pil_img.copy()
        applied = []

        if enabled:
            n = random.randint(1, min(3, len(enabled)))
            chosen = random.sample(enabled, n)

            flips = [t for t in chosen if t.startswith("flip")]
            if flips:
                flip = random.choice(flips)
                if flip == "flip_h":
                    aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                    applied.append("FlipH")
                elif flip == "flip_v":
                    aug_img = aug_img.transpose(Image.FLIP_TOP_BOTTOM)
                    applied.append("FlipV")
                elif flip == "flip_hv":
                    aug_img = aug_img.transpose(Image.ROTATE_180)
                    applied.append("FlipHV")

            if "grain" in chosen:
                aug_img = apply_camera_grain(aug_img)
                applied.append("Grain")
            if "noise" in chosen:
                if random.random() < 0.5:
                    aug_img = apply_salt_pepper_noise(aug_img, amount=random.uniform(0.008, 0.02))
                    applied.append("SaltPepper")
                else:
                    aug_img = apply_gaussian_noise(aug_img, sigma=random.uniform(10, 20))
                    applied.append("GaussNoise")
            if "blur" in chosen:
                ksize = random.choice([3, 5])
                aug_img = apply_blur(aug_img, ksize)
                applied.append(f"BlurK{ksize}")
        
        results.append((aug_img, f"_aug_{k+1}"))
        gc.collect()
        
    return results
