from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
import models
from routers import auth, users, projects, classes, images, prelabels, labels, dataset
import os

Base.metadata.create_all(bind=engine)

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
app.include_router(prelabels.router, tags=["prelabels"])
app.include_router(labels.router, tags=["labels"])
app.include_router(dataset.router, tags=["dataset"])

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
