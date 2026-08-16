from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func, Boolean
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    statut = Column(String, nullable=False, default="activated")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # 'Yolo' | 'Yolo OBB' | 'Ocr' | 'Classification' | 'Deskewer' | 'KIE'
    ocr_charset = Column(String, nullable=True)
    dbnet_model_path = Column(String, nullable=True)
    ocr_enable_class = Column(Boolean, nullable=True, default=False)
    model_img_h = Column(Integer, nullable=True)
    model_img_w = Column(Integer, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=func.now())
    statut = Column(String, nullable=False, default="activated")


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    code = Column(Integer, nullable=False)
    label = Column(String, nullable=False)
    color = Column(String, nullable=False)


class YoloLabel(Base):
    __tablename__ = "yolo_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    class_code = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)
    coordinates = Column(String, nullable=False)
    box_image = Column(String, nullable=True)


class YoloPrelabel(Base):
    __tablename__ = "yolo_prelabels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    class_code = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)
    coordinates = Column(String, nullable=False)
    box_image = Column(String, nullable=True)


class ClassificationLabel(Base):
    __tablename__ = "classification_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    class_code = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)


class ClassificationPrelabel(Base):
    __tablename__ = "classification_prelabels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    class_code = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)


class OcrLabel(Base):
    __tablename__ = "ocr_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    img_name = Column(String, nullable=False)
    value = Column(String, nullable=False)
    class_code = Column(Integer, nullable=True, default=-1)


class OcrPrelabel(Base):
    __tablename__ = "ocr_prelabels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    img_name = Column(String, nullable=False)
    value = Column(String, nullable=False)
    class_code = Column(Integer, nullable=True, default=-1)


class DeskewerLabel(Base):
    __tablename__ = "deskewer_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    img_name = Column(String, nullable=False)
    angle = Column(Integer, nullable=False)
    crop_box = Column(String, nullable=True)


class DeskewerPrelabel(Base):
    __tablename__ = "deskewer_prelabels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    img_name = Column(String, nullable=False)
    angle = Column(Integer, nullable=False)
    crop_box = Column(String, nullable=True)


class SkippedImage(Base):
    __tablename__ = "skipped_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    img_name = Column(String, nullable=False)


class KIELabel(Base):
    __tablename__ = "kie_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    class_code = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)
    coordinates = Column(String, nullable=False)
    text_value = Column(String, nullable=False)
    box_image = Column(String, nullable=True)


class KIEPrelabel(Base):
    __tablename__ = "kie_prelabels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    class_code = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)
    coordinates = Column(String, nullable=False)
    text_value = Column(String, nullable=False)
    box_image = Column(String, nullable=True)

