"""Test OBB crop on b47e0cea-BIOMETRIQUE_00066 with EXIF rotation."""
import sqlite3, numpy as np, cv2, math
from PIL import Image, ImageOps
from pathlib import Path

conn = sqlite3.connect("data/annotator.db")
c = conn.cursor()
c.execute("SELECT class_code, coordinates FROM yolo_labels WHERE project_id=8 AND img_name='b47e0cea-BIOMETRIQUE_00066.jpg' AND coordinates != ''")
labels = c.fetchall()
conn.close()

img_paths = list(Path("data/tmp_uploads").rglob("b47e0cea-BIOMETRIQUE_00066*"))
img_path = img_paths[0]

# APPLY EXIF TRANSPOSE
raw_img = Image.open(img_path)
pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
img_w, img_h = pil_img.size
np_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
print(f"Image size after EXIF rotation: {img_w}x{img_h}")

src_vis = np_bgr.copy()
for cc, coords_str in labels:
    coords = [float(x) for x in coords_str.split()]
    pts = np.array([(coords[i]*img_w, coords[i+1]*img_h) for i in range(0,8,2)], dtype="float32")

    cv2.polylines(src_vis, [pts.astype(np.int32)], True, (0,0,255), 3)
    for i, pt in enumerate(pts.astype(np.int32)):
        cv2.circle(src_vis, tuple(pt), 6, (0,255,255), -1)
        cv2.putText(src_vis, str(i), (pt[0]+4, pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    p0, p1, p2, p3 = pts
    out_w = max(1, int(round(math.hypot(float(p1[0]-p0[0]), float(p1[1]-p0[1])))))
    out_h = max(1, int(round(math.hypot(float(p2[0]-p1[0]), float(p2[1]-p1[1])))))
    dst_p = np.array([[0,0],[out_w,0],[out_w,out_h],[0,out_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst_p)
    crop = cv2.warpPerspective(np_bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
    cv2.imwrite(f"bio66det_c{cc}_exif_crop.jpg", crop)
    print(f"  class={cc} -> crop {crop.shape}")

cv2.imwrite("bio66det_exif_source.jpg", src_vis)
print("\nDone - check bio66det_exif_*.jpg")
