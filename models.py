from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func
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
    type = Column(String, nullable=False)  # 'Yolo' | 'Yolo OBB' | 'Ocr' | 'Classification'
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


class YoloPrelabel(Base):
    __tablename__ = "yolo_prelabels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    class_code = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)
    coordinates = Column(String, nullable=False)


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


class OcrPrelabel(Base):
    __tablename__ = "ocr_prelabels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    img_name = Column(String, nullable=False)
    value = Column(String, nullable=False)
