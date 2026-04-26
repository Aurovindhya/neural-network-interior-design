from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from src.config import get_settings
from enum import Enum

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


class UserRole(str, Enum):
    HR = "hr"
    PATIENT = "patient"


# Mock user DB — replace with real DB in production
MOCK_USERS = {
    "hr_user": {
        "username": "hr_user",
        "hashed_password": pwd_context.hash("hrpass123"),
        "role": UserRole.HR,
        "name": "Sarah Chen",
        "employee_id": "EMP001",
    },
    "patient_user": {
        "username": "patient_user",
        "hashed_password": pwd_context.hash("patientpass123"),
        "role": UserRole.PATIENT,
        "name": "James Okafor",
        "patient_id": "PAT001",
    },
}


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = MOCK_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.jwt_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
