from sqlalchemy.orm import Session
from datetime import datetime

from models import User
from auth import hash_password, verify_password


# -------------------- USER OPERATIONS -------------------- #

def create_user(db: Session, username: str, password: str, email: str = None) -> User:
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise ValueError("User already exists")
    user = User(
        username=username,
        password_hash=hash_password(password),
        email=email,
        created_at=datetime.utcnow()
    )
    db.add(user)
    db.flush()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.password_hash):
        return None
    update_last_login(db, user.id)
    return user


def update_last_login(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = datetime.utcnow()
