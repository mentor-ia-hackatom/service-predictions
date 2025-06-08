from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from sqlalchemy.pool import NullPool

# Construir la URL de conexión de PostgreSQL para Supabase
DATABASE_URL = f"postgresql://postgres.fctgvvwulebrcyyugawl:{settings.SUPABASE_PASSWORD}@aws-0-us-east-2.pooler.supabase.com:5432/postgres"

# Configuración específica para PostgreSQL (Supabase)
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    connect_args={
        "application_name": "service_predictions",
        "options": "-c statement_timeout=60000"
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 