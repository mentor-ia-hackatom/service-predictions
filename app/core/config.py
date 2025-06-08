from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Prediction Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Configuración de Supabase
    SUPABASE_URL: str
    SUPABASE_PASSWORD: str 
    SUPABASE_DB_URL: Optional[str] = None
    
    # Configuración de autenticación
    API_AUTH_URL: str
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 