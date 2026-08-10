import os
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
from dependencies import get_current_user
import models
from inference_service import run_yolo_inference_stream, run_ocr_inference_stream

router = APIRouter()

# Directory to temporarily store uploaded ONNX models
TMP_MODEL_DIR = Path("tmp_uploads/models")
TMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/{project_id}/auto-annotate")
async def auto_annotate(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not file.filename.endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Only .onnx models are supported")

    # Save the file temporarily
    model_path = TMP_MODEL_DIR / f"project_{project_id}_{file.filename}"
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Determine which background task to run based on project type
    if project.type in ["Yolo", "Yolo OBB"]:
        return StreamingResponse(run_yolo_inference_stream(project_id, str(model_path)), media_type="text/event-stream")
    elif project.type == "Ocr":
        return StreamingResponse(run_ocr_inference_stream(project_id, str(model_path)), media_type="text/event-stream")
    else:
        # For Classification or unsupported types
        if model_path.exists():
            os.remove(model_path)
        raise HTTPException(status_code=400, detail=f"Auto-annotation not yet supported for project type: {project.type}")


