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
            "valid": shuffled[train_c:train_c + val_c],
            "test": shuffled[train_c + val_c:],
        }

    # 3. Generate Zip
    resample_filter = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )

    zip_target = output_path if output_path else io.BytesIO()

    with zipfile.ZipFile(zip_target, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

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
                if project.type in ["Yolo", "Yolo OBB"]:
                    img_name, img_labels = item
                elif project.type == "Classification":
                    img_name, class_code = item
                elif project.type == "Ocr":
                    img_name, value = item
                elif project.type == "Deskewer":
                    img_name, (angle, crop_box) = item

                img_path = session_dir / img_name
                if not img_path.exists():
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

                            if options.augmentation and split_name in ["train", ""]:
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
                            _write_to_zip(zf, project, pil_img, prefix, stem, img_name, item, class_map)

                        # Augmentations: only for training split
                        if options.augmentation and split_name in ["train", ""]:
                            if project.type in ["Yolo", "Yolo OBB"]:
                                aug_results = augment_image_and_labels(
                                    pil_img, img_labels, project.type, options.augmentation
                                )
                                for aug_img, aug_lbls, suffix in aug_results:
                                    aug_item = (f"{stem}{suffix}.jpg", aug_lbls)
                                    _write_to_zip(
                                        zf, project, aug_img, prefix,
                                        f"{stem}{suffix}", f"{stem}{suffix}.jpg",
                                        aug_item, class_map,
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
                                    )
                                    del aug_img
                                del aug_results

                    del pil_img
                    gc.collect()

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

        # Add data.yaml for YOLO projects in full export mode
        if project.type in ["Yolo", "Yolo OBB"] and not is_crop_mode:
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


def _write_to_zip(zf, project, pil_img, prefix, stem, img_name, item, class_map):
    """Encode *pil_img* to JPEG bytes and write image + label into the zip."""
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="JPEG", quality=95)
    img_bytes = img_byte_arr.getvalue()
    img_byte_arr.close()   # free the intermediate buffer immediately

    if project.type in ["Yolo", "Yolo OBB"]:
        img_labels = item[1]
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
        value = item[1]
        zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)
        zf.writestr(f"{prefix}labels/{stem}.txt", value)

    elif project.type == "Deskewer":
        # Deskewer projects only export the deskewed images; label files are not generated.
        zf.writestr(f"{prefix}images/{stem}.jpg", img_bytes)

