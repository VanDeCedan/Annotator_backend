from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
import models
from routers import auth, users, projects, classes, images, prelabels, labels, dataset, inference
import os

from sqlalchemy import text

Base.metadata.create_all(bind=engine)

def auto_migrate_schema():
    with engine.connect() as conn:
        for model in [models.YoloLabel, models.YoloPrelabel, models.Project, models.OcrLabel, models.OcrPrelabel]:
            table_name = model.__tablename__
            try:
                res = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                existing_cols = {row[1] for row in res}
                if existing_cols:
                    for col_name, col_obj in model.__table__.columns.items():
                        if col_name not in existing_cols:
                            col_type = "BOOLEAN" if col_name == "ocr_enable_class" else "INTEGER" if col_name == "class_code" else "VARCHAR"
                            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
                            conn.commit()
            except Exception as e:
                print(f"Auto-migration info for {table_name}: {e}")

auto_migrate_schema()

app = FastAPI(title="CV Annotator API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(projects.router, prefix="/projects", tags=["projects"])
app.include_router(classes.router, tags=["classes"])
app.include_router(images.router, tags=["images"])
app.include_router(images.box_images_router, tags=["box_images"])
app.include_router(prelabels.router, tags=["prelabels"])
app.include_router(labels.router, tags=["labels"])
app.include_router(dataset.router, tags=["dataset"])
app.include_router(inference.router, prefix="/projects", tags=["inference"])

@app.on_event("startup")
def create_initial_admin():
    from database import SessionLocal
    from auth import hash_password
    db = SessionLocal()
    try:
        if db.query(models.User).count() == 0:
            admin = models.User(
                name="Administrator",
                username="admin",
                password=hash_password("admin123"),
                role="admin",
                statut="activated"
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()
