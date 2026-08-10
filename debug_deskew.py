import cv2
from PIL import Image, ImageOps

img_path = "data/tmp_uploads/9/local_workspace/114cc15c-BIOMETRIQUE_00098_crop_1.jpg"
angle = -90

# What dataset generator does:
with Image.open(img_path) as raw_img:
    pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    
    # Save original to see how it looks
    pil_img.save("debug_114_orig.jpg")
    
    # Rotate according to dataset generator
    resample = Image.BICUBIC
    rotated_img = pil_img.rotate(angle, expand=True, resample=resample)
    rotated_img.save("debug_114_rotated_backend.jpg")

print("Done. Generated debug_114_orig.jpg and debug_114_rotated_backend.jpg")
