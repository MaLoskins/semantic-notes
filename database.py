import time
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=15,
    max_overflow=30,
    future=True,
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

log = logging.getLogger("database")
logging.basicConfig(level=logging.INFO)

def _wait_for_db(max_attempts: int = 30, delay_seconds: float = 1.0) -> None:
    """Poll the DB until it accepts TCP connections."""
    attempt = 0
    last_err: Exception | None = None
    while attempt < max_attempts:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("Database connection OK.")
            return
        except OperationalError as e:
            last_err = e
            attempt += 1
            time.sleep(delay_seconds)
    # Exhausted retries
    raise OperationalError(f"Could not connect to database after {max_attempts} attempts", params=None, orig=last_err)

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Initialize database tables
def init_db():
    try:
        _wait_for_db()
        Base.metadata.create_all(bind=engine)
        log.info("Database initialized and tables created successfully.")
    except Exception as e:
        log.error(f"Database initialization failed: {e}")
        raise
