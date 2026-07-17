import io
import os
import zipfile
import random
from pathlib import Path
from PIL import Image
import models
from schemas import DatasetRequest
from augmentation import augment_image_and_labels, augment_image_only
from routers.images import UPLOAD_DIR

def generate_dataset_zip(
    project: models.Project,
    session_id: str,
    labels_data: list,
    class_map: dict,
    options: DatasetRequest
) -> io.BytesIO:
    
    session_dir = UPLOAD_DIR / str(project.id) / session_id
    
    zip_io = io.BytesIO()
    
    # 1. Parse resize
    target_size = None
    if options.resize:
        try:
            parts = options.resize.lower().split("x")
            if len(parts) == 2:
                target_size = (int(parts[0].strip()), int(parts[1].strip()))
        except Exception:
            pass

    # 2. Split
    splits = {"": labels_data}
    if options.split_enabled:
        total = options.train_pct + options.val_pct + options.test_pct
        if abs(total - 100.0) > 0.01:
            train_pct = (options.train_pct / total) * 100.0
            val_pct = (options.val_pct / total) * 100.0
        else:
            train_pct = options.train_pct
            val_pct = options.val_pct
            
        shuffled = list(labels_data)
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_c = int(round(n * (train_pct / 100.0)))
        val_c = int(round(n * (val_pct / 100.0)))
        
        splits = {
            "train": shuffled[:train_c],
            "valid": shuffled[train_c:train_c+val_c],
            "test": shuffled[train_c+val_c:]
        }
    
    # 3. Generate Zip
    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        
        for split_name, split_data in splits.items():
            if not split_data: continue
            
            prefix = f"{split_name}/" if split_name else ""
            
            for item in split_data:
                # Unpack
                if project.type in ["Yolo", "Yolo OBB"]:
                    img_name, img_labels = item
                elif project.type == "Classification":
                    img_name, class_code = item
                elif project.type == "Ocr":
                    img_name, value = item
                
                # Load image
                img_path = session_dir / img_name
                if not img_path.exists(): continue
                
                try:
                    with Image.open(img_path).convert("RGB") as pil_img:
                        if target_size:
                            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                            pil_img = pil_img.resize(target_size, resample_filter)
                            
                        stem = Path(img_name).stem
                        
                        # Process and write original
                        _write_to_zip(zf, project, pil_img, prefix, stem, img_name, item, class_map)
                        
                        # Augmentations (only apply to train set or when no split)
                        if options.augmentation and split_name in ["train", ""]:
                            if project.type in ["Yolo", "Yolo OBB"]:
                                aug_results = augment_image_and_labels(pil_img, img_labels, project.type, options.augmentation)
                                for aug_img, aug_lbls, suffix in aug_results:
                                    aug_item = (f"{stem}{suffix}.jpg", aug_lbls)
                                    _write_to_zip(zf, project, aug_img, prefix, f"{stem}{suffix}", f"{stem}{suffix}.jpg", aug_item, class_map)
                            else:
                                aug_results = augment_image_only(pil_img, options.augmentation)
                                for aug_img, suffix in aug_results:
                                    if project.type == "Classification":
                                        aug_item = (f"{stem}{suffix}.jpg", class_code)
                                    else:
                                        aug_item = (f"{stem}{suffix}.jpg", value)
                                    _write_to_zip(zf, project, aug_img, prefix, f"{stem}{suffix}", f"{stem}{suffix}.jpg", aug_item, class_map)

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
                    
        # Add data.yaml for YOLO
        if project.type in ["Yolo", "Yolo OBB"]:
            classes_list = [class_map[k] for k in sorted(class_map.keys())]
            names_str = "[" + ", ".join([f"'{n}'" for n in classes_list]) + "]"
            
            if options.yolo_version == "v5" or options.yolo_version == "v8":
                 yaml_content = f"""path: ./
train: train/images
val: valid/images
test: test/images
nc: {len(classes_list)}
names: {names_str}
"""
            else: # v11 or others
                 yaml_content = f"""path: ./
train: train/images
val: valid/images
test: test/images
nc: {len(classes_list)}
names: {names_str}
"""
            zf.writestr("data.yaml", yaml_content)

    return zip_io

def _write_to_zip(zf, project, pil_img, prefix, stem, img_name, item, class_map):
    # Save Image to bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG', quality=95)
    img_bytes = img_byte_arr.getvalue()
    
    if project.type in ["Yolo", "Yolo OBB"]:
        img_labels = item[1]
        zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
        
        lbl_content = ""
        for c_code, coords in img_labels:
            coords_str = " ".join(f"{c:.6f}" for c in coords)
            lbl_content += f"{c_code} {coords_str}\n"
        zf.writestr(f"{prefix}labels/{stem}.txt", lbl_content)
        
    elif project.type == "Classification":
        class_code = item[1]
        class_name = class_map.get(class_code, f"class_{class_code}")
        # Sanitize folder name
        class_name = class_name.replace(" ", "_").replace("/", "_")
        zf.writestr(f"{prefix}{class_name}/{stem}.jpg", img_bytes)
        
    elif project.type == "Ocr":
        value = item[1]
        zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
        zf.writestr(f"{prefix}labels/{stem}.txt", value)
