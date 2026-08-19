import io
import os
import zipfile
import random
import gc
from pathlib import Path
from PIL import Image
import models
from schemas import DatasetRequest
from augmentation import augment_image_and_labels, augment_image_only
from routers.images import UPLOAD_DIR
import json

import math

def composite_box_images(pil_img, img_labels, project_id, project_type):
    if not img_labels:
        return pil_img
    
    box_dir = UPLOAD_DIR / str(project_id) / "box_images"
    if not box_dir.exists():
        return pil_img

    img_w, img_h = pil_img.size
    
    for item in img_labels:
        box_image_name = item[2] if len(item) > 2 else None
        if not box_image_name:
            continue
            
        box_path = box_dir / box_image_name
        if not box_path.exists():
            continue
            
        try:
            with Image.open(box_path) as box_raw:
                box_img = box_raw.convert("RGBA")
                coords = item[1]
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                bicubic = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
                
                if project_type == "Yolo OBB" or len(coords) == 8:
                    if len(coords) == 8:
                        x1, y1, x2, y2, x3, y3, x4, y4 = [
                            coords[0] * img_w, coords[1] * img_h,
                            coords[2] * img_w, coords[3] * img_h,
                            coords[4] * img_w, coords[5] * img_h,
                            coords[6] * img_w, coords[7] * img_h
                        ]
                        cx = (x1 + x2 + x3 + x4) / 4.0
                        cy = (y1 + y2 + y3 + y4) / 4.0
                        w = math.hypot(x2 - x1, y2 - y1)
                        h = math.hypot(x3 - x2, y3 - y2)
                        angle_rad = math.atan2(y2 - y1, x2 - x1)
                        angle_deg = math.degrees(angle_rad)
                        
                        box_resized = box_img.resize((max(1, int(round(w))), max(1, int(round(h)))), resample)
                        box_rotated = box_resized.rotate(-angle_deg, expand=True, resample=bicubic)
                        
                        paste_x = int(round(cx - box_rotated.width / 2.0))
                        paste_y = int(round(cy - box_rotated.height / 2.0))
                        
                        pil_img.paste(box_rotated, (paste_x, paste_y), box_rotated)
                elif len(coords) == 4:
                    cx, cy, w, h = coords[:4]
                    bw = int(round(w * img_w))
                    bh = int(round(h * img_h))
                    if bw > 0 and bh > 0:
                        box_resized = box_img.resize((bw, bh), resample)
                        top_left_x = int(round((cx - w / 2.0) * img_w))
                        top_left_y = int(round((cy - h / 2.0) * img_h))
                        pil_img.paste(box_resized, (top_left_x, top_left_y), box_resized)
        except Exception as e:
            print(f"Error compositing box image {box_image_name}: {e}")
            
    return pil_img


def generate_dataset_zip(
    project: models.Project,
    session_id: str,
    labels_data: list,
    class_map: dict,
    options: DatasetRequest,
    progress_callback=None,
    output_path: str = None,
) -> None:

    session_dir = UPLOAD_DIR / str(project.id) / session_id

    export_mode = getattr(options, "export_mode", "full") or "full"
    is_crop_mode = export_mode in ["crop", "sliced"] and project.type in ["Yolo", "Yolo OBB"]

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
    is_dbnet_format = (
        (project.type == "KIE" and getattr(options, "kie_export_format", "dbnet") == "dbnet") or
        (project.type in ["Yolo", "Yolo OBB"] and getattr(options, "yolo_export_format", "yolo") == "dbnet")
    )

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

        val_key = "val" if is_dbnet_format else "valid"

        splits = {
            "train": shuffled[:train_c],
            val_key: shuffled[train_c:train_c + val_c],
            "test": shuffled[train_c + val_c:],
        }

    # 3. Generate Zip
    resample_filter = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )

    zip_target = output_path if output_path else io.BytesIO()

    with zipfile.ZipFile(zip_target, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        split_images_collected = {}
        vit_rows_collected = {}
        ner_data_collected = {}

        total_items = sum(len(sd) for sd in splits.values())
        current_item = 0

        for split_name, split_data in splits.items():
            if not split_data:
                continue

            prefix = f"{split_name}/" if split_name else ""

            for item in split_data:
                current_item += 1
                if progress_callback:
                    progress_callback(current_item, total_items)

                # Unpack per project type
                if project.type in ["Yolo", "Yolo OBB", "KIE", "NER"]:
                    img_name, img_labels = item
                elif project.type == "Classification":
                    img_name, class_code = item
                elif project.type == "Ocr":
                    img_name, ocr_val = item
                    if isinstance(ocr_val, tuple):
                        value, class_code = ocr_val
                    else:
                        value, class_code = ocr_val, -1
                    item = (img_name, (value, class_code))
                elif project.type == "Deskewer":
                    img_name, (angle, crop_box) = item

                img_path = session_dir / img_name
                if not img_path.exists():
                    continue

                if project.type == "NER":
                    with open(img_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    
                    entities = []
                    for lbl in img_labels:
                        c_code, start_char, end_char, _ = lbl
                        class_name = class_map.get(c_code, f"class_{c_code}")
                        entities.append({
                            "start": start_char,
                            "end": end_char,
                            "label": class_name
                        })
                    
                    if split_name not in ner_data_collected:
                        ner_data_collected[split_name] = []
                        
                    ner_data_collected[split_name].append({
                        "text": text_content,
                        "entities": entities
                    })
                    continue

                try:
                    with Image.open(img_path) as raw_img:
                        pil_img = raw_img.convert("RGB")
                        if project.type in ["Yolo", "Yolo OBB"]:
                            pil_img = composite_box_images(pil_img, img_labels, project.id, project.type)
                        elif project.type == "Deskewer":
                            # Crop the original image first using the crop_box coordinates
                            if crop_box:
                                try:
                                    x, y, w, h = map(float, crop_box.split(','))
                                    img_w, img_h = pil_img.size
                                    left = int(round(x * img_w))
                                    top = int(round(y * img_h))
                                    right = int(round((x + w) * img_w))
                                    bottom = int(round((y + h) * img_h))
                                    left = max(0, min(left, img_w))
                                    top = max(0, min(top, img_h))
                                    right = max(0, min(right, img_w))
                                    bottom = max(0, min(bottom, img_h))
                                    if right > left and bottom > top:
                                        pil_img = pil_img.crop((left, top, right, bottom))
                                except Exception as crop_err:
                                    print(f"Error cropping deskewer image {img_name}: {crop_err}")
                            # Rotate the image to straighten it.
                            # CSS rotate(+Ndeg) is clockwise; PIL.rotate(+N) is counter-clockwise,
                            # so we negate the angle to match what the annotator saw on screen.
                            if angle != 0:
                                resample_filter_rot = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
                                pil_img = pil_img.rotate(-angle, expand=True, resample=resample_filter_rot)
                        if getattr(options, "grayscale", False):
                            pil_img = pil_img.convert("L").convert("RGB")
                        if target_size and not is_crop_mode:
                            pil_img = pil_img.resize(target_size, resample_filter)

                    stem = Path(img_name).stem

                    if is_crop_mode:
                        # Slice boxes from image
                        img_w, img_h = pil_img.size
                        for idx, lbl in enumerate(img_labels):
                            c_code = lbl[0]
                            coords = lbl[1]
                            if not coords:
                                continue
                            if project.type == "Yolo OBB" or len(coords) == 8:
                                x_pts = [c * img_w for c in coords[0::2]]
                                y_pts = [c * img_h for c in coords[1::2]]
                                x_min = max(0, int(round(min(x_pts))))
                                y_min = max(0, int(round(min(y_pts))))
                                x_max = min(img_w, int(round(max(x_pts))))
                                y_max = min(img_h, int(round(max(y_pts))))
                            else:
                                cx, cy, w, h = coords[:4]
                                x_min = max(0, int(round((cx - w / 2.0) * img_w)))
                                y_min = max(0, int(round((cy - h / 2.0) * img_h)))
                                x_max = min(img_w, int(round((cx + w / 2.0) * img_w)))
                                y_max = min(img_h, int(round((cy + h / 2.0) * img_h)))

                            if x_max <= x_min or y_max <= y_min:
                                continue

                            crop_img = pil_img.crop((x_min, y_min, x_max, y_max))
                            if target_size:
                                crop_img = crop_img.resize(target_size, resample_filter)

                            class_name = class_map.get(c_code, f"class_{c_code}")
                            class_name = str(class_name).replace(" ", "_").replace("/", "_")
                            crop_stem = f"{stem}_crop_{idx + 1}"

                            _write_single_image_to_zip(
                                zf, crop_img, f"{prefix}{class_name}/{crop_stem}.jpg"
                            )

                            if options.augmentation and (split_name in ["train", ""] or (split_name in ["val", "valid"] and getattr(options.augmentation, "include_aug_in_val", False))):
                                aug_results = augment_image_only(crop_img, options.augmentation)
                                for aug_img, suffix in aug_results:
                                    _write_single_image_to_zip(
                                        zf, aug_img, f"{prefix}{class_name}/{crop_stem}{suffix}.jpg"
                                    )
                                    del aug_img
                                del aug_results

                            del crop_img

                    else:
                        # Write original image + labels
                        if project.type == "Deskewer":
                            deskewed_item = (img_name, (0, None))
                            _write_to_zip(zf, project, pil_img, prefix, stem, img_name, deskewed_item, class_map)
                        else:
                            _write_to_zip(zf, project, pil_img, prefix, stem, img_name, item, class_map, options, split_name, split_images_collected, vit_rows_collected)

                        # Augmentations: only for training split
                        if options.augmentation and (split_name in ["train", ""] or (split_name in ["val", "valid"] and getattr(options.augmentation, "include_aug_in_val", False))):
                            if project.type in ["Yolo", "Yolo OBB", "KIE"]:
                                aug_results = augment_image_and_labels(
                                    pil_img, img_labels, project.type, options.augmentation
                                )
                                for aug_img, aug_lbls, suffix in aug_results:
                                    aug_item = (f"{stem}{suffix}.jpg", aug_lbls)
                                    _write_to_zip(
                                        zf, project, aug_img, prefix,
                                        f"{stem}{suffix}", f"{stem}{suffix}.jpg",
                                        aug_item, class_map,
                                        options, split_name, split_images_collected
                                    )
                                    del aug_img
                                del aug_results
                            elif project.type == "Deskewer":
                                deskew_angles = getattr(options.augmentation, "deskew_angles", [])
                                if deskew_angles:
                                    resample_filter = (
                                        Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
                                    )
                                    for idx, da in enumerate(deskew_angles):
                                        aug_img = pil_img.rotate(da, expand=True, resample=resample_filter)
                                        # Since pil_img is already deskewed (straight), the new straightening angle is -da
                                        new_angle = (-da + 180) % 360 - 180
                                        aug_item = (f"{stem}_aug_{idx+1}.jpg", (new_angle, None))
                                        _write_to_zip(
                                            zf, project, aug_img, prefix,
                                            f"{stem}_aug_{idx+1}", f"{stem}_aug_{idx+1}.jpg",
                                            aug_item, class_map,
                                        )
                                        del aug_img
                                else:
                                    aug_results = augment_image_only(pil_img, options.augmentation)
                                    for aug_img, suffix in aug_results:
                                        aug_item = (f"{stem}{suffix}.jpg", (0, None))
                                        _write_to_zip(
                                            zf, project, aug_img, prefix,
                                            f"{stem}{suffix}", f"{stem}{suffix}.jpg",
                                            aug_item, class_map,
                                        )
                                        del aug_img
                                    del aug_results
                            else:
                                aug_results = augment_image_only(pil_img, options.augmentation)
                                for aug_img, suffix in aug_results:
                                    if project.type == "Classification":
                                        aug_item = (f"{stem}{suffix}.jpg", class_code)
                                    else:
                                        aug_item = (f"{stem}{suffix}.jpg", value)
                                    _write_to_zip(
                                        zf, project, aug_img, prefix,
                                        f"{stem}{suffix}", f"{stem}{suffix}.jpg",
                                        aug_item, class_map,
                                        options, split_name, split_images_collected, vit_rows_collected
                                    )
                                    del aug_img
                                del aug_results

                    del pil_img
                    gc.collect()

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

        # Add data.yaml for YOLO projects in full export mode (except for DBNet format)
        if project.type in ["Yolo", "Yolo OBB"] and not is_crop_mode and not is_dbnet_format:
            classes_list = [class_map[k] for k in sorted(class_map.keys())]
            names_str = "[" + ", ".join([f"'{n}'" for n in classes_list]) + "]"

            yaml_content = (
                f"path: ./\n"
                f"train: train/images\n"
                f"val: valid/images\n"
                f"test: test/images\n"
                f"nc: {len(classes_list)}\n"
                f"names: {names_str}\n"
            )
            zf.writestr("data.yaml", yaml_content)

        # Write list files if (KIE and format is dbnet) or (Yolo/Yolo OBB and format is dbnet)
        if is_dbnet_format and split_images_collected:
            for split_name, images in split_images_collected.items():
                if not images:
                    continue
                disp_split = "val" if split_name in ["valid", "val"] else split_name
                prefix = f"{split_name}/" if split_name else ""
                
                # Write prefix/db_{disp_split}_list.txt or db_dataset_list.txt
                list_name = f"db_{disp_split}_list.txt" if disp_split else "db_dataset_list.txt"
                list_content = "\n".join(f"images/{img}" for img in images)
                zf.writestr(f"{prefix}{list_name}", list_content)

        # Write ViT CSV files if OCR and vit format
        if project.type == "Ocr" and getattr(options, "ocr_export_format", "ocr") == "vit" and vit_rows_collected:
            import csv
            for split_name, rows in vit_rows_collected.items():
                if not rows:
                    continue
                prefix = f"{split_name}/" if split_name else ""
                csv_io = io.StringIO()
                writer = csv.writer(csv_io)
                # Header
                writer.writerow(["chemin_image_decoupee", "texte", "classe"])
                for row in rows:
                    writer.writerow(row)
                
                # Write csv to zip
                csv_filename = f"{prefix}dataset.csv"
                zf.writestr(csv_filename, csv_io.getvalue())
                csv_io.close()

        # Write NER JSON
        if project.type == "NER" and ner_data_collected:
            for split_name, ner_docs in ner_data_collected.items():
                if not ner_docs:
                    continue
                prefix = f"{split_name}/" if split_name else ""
                json_content = json.dumps(ner_docs, indent=2, ensure_ascii=False)
                zf.writestr(f"{prefix}dataset.json", json_content.encode("utf-8"))

    # File is closed when 'with' block exits.
    if output_path is None:
        zip_target.seek(0)
        return zip_target



def _write_single_image_to_zip(zf, pil_img, zip_path):
    """Encode *pil_img* to JPEG bytes and write directly to zip_path."""
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="JPEG", quality=95)
    zf.writestr(zip_path, img_byte_arr.getvalue())
    img_byte_arr.close()


def _write_to_zip(zf, project, pil_img, prefix, stem, img_name, item, class_map, options=None, split_name="", split_images_collected=None, vit_rows_collected=None):
    """Encode *pil_img* to JPEG bytes and write image + label into the zip."""
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="JPEG", quality=95)
    img_bytes = img_byte_arr.getvalue()
    img_byte_arr.close()   # free the intermediate buffer immediately

    if project.type in ["Yolo", "Yolo OBB"]:
        img_labels = item[1]
        if getattr(options, "yolo_export_format", "yolo") == "dbnet":
            zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
            img_w, img_h = pil_img.size
            lines = []
            for lbl in img_labels:
                c_code = lbl[0]
                coords = lbl[1]
                if len(coords) >= 4:
                    cx, cy, w, h = coords[:4]
                    x_min = int(round((cx - w / 2.0) * img_w))
                    y_min = int(round((cy - h / 2.0) * img_h))
                    x_max = int(round((cx + w / 2.0) * img_w))
                    y_max = int(round((cy + h / 2.0) * img_h))
                    
                    x_min = max(0, min(x_min, img_w))
                    y_min = max(0, min(y_min, img_h))
                    x_max = max(0, min(x_max, img_w))
                    y_max = max(0, min(y_max, img_h))
                    
                    class_name = class_map.get(c_code, f"class_{c_code}")
                    lines.append(f"{x_min},{y_min},{x_max},{y_min},{x_max},{y_max},{x_min},{y_max},{class_name}")
            zf.writestr(f"{prefix}gt/{stem}.txt", "\n".join(lines))
            
            # Record for lists
            if split_images_collected is not None:
                if split_name not in split_images_collected:
                    split_images_collected[split_name] = []
                split_images_collected[split_name].append(f"{stem}.jpg")
        else:
            zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
            lbl_content = ""
            for lbl in img_labels:
                c_code = lbl[0]
                coords = lbl[1]
                coords_str = " ".join(f"{c:.6f}" for c in coords)
                lbl_content += f"{c_code} {coords_str}\n"
            zf.writestr(f"{prefix}labels/{stem}.txt", lbl_content)

    elif project.type == "Classification":
        class_code = item[1]
        class_name = class_map.get(class_code, f"class_{class_code}")
        class_name = class_name.replace(" ", "_").replace("/", "_")
        zf.writestr(f"{prefix}{class_name}/{stem}.jpg", img_bytes)

    elif project.type == "Ocr":
        if isinstance(item[1], tuple):
            value, class_code = item[1]
        else:
            value, class_code = item[1], -1
            
        if getattr(options, "ocr_export_format", "ocr") == "vit":
            zf.writestr(f"{prefix}crops/{stem}.jpg", img_bytes)
            class_name = class_map.get(class_code, f"class_{class_code}") if class_code != -1 else ""
            if vit_rows_collected is not None:
                if split_name not in vit_rows_collected:
                    vit_rows_collected[split_name] = []
                img_zip_path = f"{prefix}crops/{stem}.jpg"
                vit_rows_collected[split_name].append((img_zip_path, value, class_name))
        else:
            zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
            zf.writestr(f"{prefix}labels/{stem}.txt", value)

    elif project.type == "Deskewer":
        # Deskewer projects only export the deskewed images; label files are not generated.
        zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
        
    elif project.type == "KIE":
        img_labels = item[1]
        zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
        img_w, img_h = pil_img.size
        
        # 1. DBNet Format
        if getattr(options, "kie_export_format", "dbnet") == "dbnet":
            lines = []
            for lbl in img_labels:
                c_code = lbl[0]
                coords = lbl[1]
                if len(coords) >= 4:
                    cx, cy, w, h = coords[:4]
                    x_min = int(round((cx - w / 2.0) * img_w))
                    y_min = int(round((cy - h / 2.0) * img_h))
                    x_max = int(round((cx + w / 2.0) * img_w))
                    y_max = int(round((cy + h / 2.0) * img_h))
                    
                    x_min = max(0, min(x_min, img_w))
                    y_min = max(0, min(y_min, img_h))
                    x_max = max(0, min(x_max, img_w))
                    y_max = max(0, min(y_max, img_h))
                    
                    class_name = class_map.get(c_code, f"class_{c_code}")
                    lines.append(f"{x_min},{y_min},{x_max},{y_min},{x_max},{y_max},{x_min},{y_max},{class_name}")
            zf.writestr(f"{prefix}gt/{stem}.txt", "\n".join(lines))
            
            # Record for lists
            if split_name not in split_images_collected:
                split_images_collected[split_name] = []
            split_images_collected[split_name].append(f"{stem}.jpg")

        # 2. Spatial KIE Format
        else:
            boxes_list = []
            for idx, lbl in enumerate(img_labels):
                c_code = lbl[0]
                coords = lbl[1]
                text_val = lbl[2] if len(lbl) > 2 else ""
                class_name = class_map.get(c_code, f"class_{c_code}")
                if len(coords) >= 4:
                    cx, cy, w, h = coords[:4]
                    x_min = round(max(0.0, min(cx - w / 2.0, 1.0)), 6)
                    y_min = round(max(0.0, min(cy - h / 2.0, 1.0)), 6)
                    x_max = round(max(0.0, min(cx + w / 2.0, 1.0)), 6)
                    y_max = round(max(0.0, min(cy + h / 2.0, 1.0)), 6)
                    boxes_list.append({
                        "index": idx,
                        "class_code": c_code,
                        "class_name": class_name,
                        "text_value": text_val,
                        "cx": round(cx, 6),
                        "cy": round(cy, 6),
                        "w": round(w, 6),
                        "h": round(h, 6),
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    })
                    
            annotations = []
            for b in boxes_list:
                left_neighbor = None
                right_neighbor = None
                up_neighbor = None
                down_neighbor = None
                
                min_d_left = float('inf')
                min_d_right = float('inf')
                min_d_up = float('inf')
                min_d_down = float('inf')
                
                for other in boxes_list:
                    if other["index"] == b["index"]:
                        continue
                    
                    dx = other["cx"] - b["cx"]
                    dy = other["cy"] - b["cy"]
                    dist = (dx**2 + dy**2)**0.5
                    
                    if dx < 0 and abs(dy) <= abs(dx): # Left
                        if dist < min_d_left:
                            min_d_left = dist
                            left_neighbor = other
                    elif dx > 0 and abs(dy) <= abs(dx): # Right
                        if dist < min_d_right:
                            min_d_right = dist
                            right_neighbor = other
                    elif dy < 0 and abs(dx) < abs(dy): # Up
                        if dist < min_d_up:
                            min_d_up = dist
                            up_neighbor = other
                    elif dy > 0 and abs(dx) < abs(dy): # Down
                        if dist < min_d_down:
                            min_d_down = dist
                            down_neighbor = other
                            
                def make_relation(nb, dist):
                    if nb is None:
                        return None
                    return {
                        "box_index": nb["index"],
                        "distance": round(dist, 6),
                        "w": nb["w"],
                        "h": nb["h"],
                        "class_name": nb["class_name"]
                    }
                    
                annotations.append({
                    "box_index": b["index"],
                    "class_code": b["class_code"],
                    "class_name": b["class_name"],
                    "text_value": b["text_value"],
                    "normalized_coords": {
                        "x_min": b["x_min"],
                        "y_min": b["y_min"],
                        "x_max": b["x_max"],
                        "y_max": b["y_max"],
                        "cx": b["cx"],
                        "cy": b["cy"],
                        "w": b["w"],
                        "h": b["h"]
                    },
                    "relations": {
                        "left": make_relation(left_neighbor, min_d_left),
                        "right": make_relation(right_neighbor, min_d_right),
                        "up": make_relation(up_neighbor, min_d_up),
                        "down": make_relation(down_neighbor, min_d_down)
                    }
                })
                
            json_content = json.dumps({
                "file_name": f"{stem}.jpg",
                "image_width": img_w,
                "image_height": img_h,
                "annotations": annotations
            }, indent=2)
            zf.writestr(f"{prefix}labels/{stem}.json", json_content)


