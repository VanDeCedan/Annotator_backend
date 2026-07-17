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
    type: str  # Yolo | Yolo OBB | Ocr | Classification

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None

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

class ClassOut(ClassBase):
    id: int
    project_id: int
    code: int
    model_config = ConfigDict(from_attributes=True)

# --- Label Schemas ---
class YoloLabelItem(BaseModel):
    class_code: int
    coordinates: str

class YoloLabelRequest(BaseModel):
    img_name: str
    labels: List[YoloLabelItem]

class ClassificationLabelRequest(BaseModel):
    img_name: str
    class_code: int

class OcrLabelRequest(BaseModel):
    img_name: str
    value: str

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
    blur: bool = False
    num_augs: int = 3

class DatasetRequest(BaseModel):
    session_id: str
    resize: Optional[str] = None
    augmentation: Optional[AugmentationOptions] = None
    split_enabled: bool = False
    train_pct: float = 70.0
    val_pct: float = 20.0
    test_pct: float = 10.0
    yolo_version: str = "v8"
