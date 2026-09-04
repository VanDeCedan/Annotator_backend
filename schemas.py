from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

# --- User Schemas ---
class UserBase(BaseModel):
    name: str
    username: str
    role: str

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None

class UserOut(UserBase):
    id: int
    statut: str
    model_config = ConfigDict(from_attributes=True)

# --- Project Schemas ---
class ProjectBase(BaseModel):
    name: str
    type: str  # Yolo | Yolo OBB | Ocr | Classification | Deskewer | KIE | NER
    ocr_charset: Optional[str] = None
    dbnet_model_path: Optional[str] = None
    ocr_enable_class: Optional[bool] = False
    model_img_h: Optional[int] = None
    model_img_w: Optional[int] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    ocr_charset: Optional[str] = None
    dbnet_model_path: Optional[str] = None
    ocr_enable_class: Optional[bool] = None
    model_img_h: Optional[int] = None
    model_img_w: Optional[int] = None

class ProjectDuplicate(BaseModel):
    name: str
    duplicate_data: bool = False

class ProjectOut(ProjectBase):
    id: int
    created_by: int
    created_at: datetime
    statut: str
    model_config = ConfigDict(from_attributes=True)

# --- Class Schemas ---
class ClassBase(BaseModel):
    label: str
    color: str

class ClassCreate(ClassBase):
    pass

class ClassUpdate(BaseModel):
    label: Optional[str] = None
    color: Optional[str] = None
    code: Optional[int] = None

class ClassOut(ClassBase):
    id: int
    project_id: int
    code: int
    model_config = ConfigDict(from_attributes=True)

# --- Label Schemas ---
class YoloLabelItem(BaseModel):
    class_code: int
    coordinates: str
    box_image: Optional[str] = None

class YoloLabelRequest(BaseModel):
    img_name: str
    labels: List[YoloLabelItem]

class ClassificationLabelRequest(BaseModel):
    img_name: str
    class_code: int

class OcrLabelRequest(BaseModel):
    img_name: str
    value: str
    class_code: Optional[int] = -1

class DeskewerLabelRequest(BaseModel):
    img_name: str
    angle: int
    crop_box: Optional[str] = None

class SkipImageRequest(BaseModel):
    img_name: str

class KIELabelItem(BaseModel):
    class_code: int
    coordinates: str
    text_value: Optional[str] = ""
    box_image: Optional[str] = None

class KIELabelRequest(BaseModel):
    img_name: str
    labels: List[KIELabelItem]

class NERLabelItem(BaseModel):
    class_code: int
    start_char: int
    end_char: int
    text_value: str

class NERLabelRequest(BaseModel):
    file_name: str
    labels: List[NERLabelItem]

class VLMLabelItem(BaseModel):
    class_code: int
    text_value: str

class VLMLabelRequest(BaseModel):
    img_name: str
    labels: List[VLMLabelItem]

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# --- Dataset Export Schemas ---
class AugmentationOptions(BaseModel):
    flip_h: bool = False
    flip_v: bool = False
    flip_hv: bool = False
    grain: bool = False
    noise: bool = False
    noise_intensity: float = 5.0
    blur: bool = False
    blur_intensity: float = 5.0
    max_rotation: int = 0
    num_augs: int = 3
    deskew_angles: List[int] = []
    
    # OCR-Specific Augmentations
    ocr_distortion_intensity: float = 0.0
    ocr_noise_intensity: float = 0.0
    ocr_blur_intensity: float = 0.0
    include_aug_in_val: bool = False

class DatasetRequest(BaseModel):
    session_id: str
    task_id: Optional[str] = None
    export_mode: str = "full"  # "full" | "crop" (or "sliced")
    resize: Optional[str] = None
    grayscale: bool = False
    augmentation: Optional[AugmentationOptions] = None
    split_enabled: bool = False
    train_pct: float = 70.0
    val_pct: float = 20.0
    test_pct: float = 10.0
    yolo_version: str = "v8"
    kie_export_format: Optional[str] = "dbnet"
    yolo_export_format: Optional[str] = "yolo"  # "yolo" | "dbnet"
    ocr_export_format: Optional[str] = "ocr"   # "ocr" | "vit"
    vlm_export_format: Optional[str] = "smolvlm"  # "smolvlm" | "donut" | "moondream2"


