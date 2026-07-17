from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import get_db
from auth import decode_token
import models

bearer = HTTPBearer()

def get_current_user(
    db: Session = Depends(get_db)
) -> models.User:
    # Bypass authentication for now, always return admin
    user = db.query(models.User).filter(models.User.username == "admin").first()
    if not user:
        user = models.User(id=1, name="Admin", username="admin", role="admin", statut="activated")
    return user

def require_role(*roles: str):
    def checker(current_user: models.User = Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user
    return checker
