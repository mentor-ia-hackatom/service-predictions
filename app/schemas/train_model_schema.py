from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class StudentData(BaseModel):
    # Características de rendimiento
    avg_grade: float = Field(..., description="Promedio de calificaciones del estudiante")
    grade_stddev: float = Field(..., description="Desviación estándar de las calificaciones")
    retry_rate: float = Field(..., description="Tasa de intentos de tareas")
    avg_delivery_delay_days: float = Field(..., description="Promedio de días de retraso en entregas")
    avg_submission_time_diff_hours: float = Field(..., description="Promedio de horas entre envíos")

    # Características de participación
    total_login_time_hours: float = Field(..., description="Tiempo total de inicio de sesión en horas")
    classes_attended: int = Field(..., description="Número de clases asistidas")
    classes_missed: int = Field(..., description="Número de clases perdidas")
    last_login_days_ago: float = Field(..., description="Días desde el último inicio de sesión")
    resource_interactions: int = Field(..., description="Número de interacciones con recursos")

    # Características de entrega
    late_submissions_count: int = Field(..., description="Número de entregas tardías")
    missing_tasks: int = Field(..., description="Número de tareas faltantes")
    completed_tasks: int = Field(..., description="Número de tareas completadas")
    total_tasks: int = Field(..., description="Número total de tareas")

    # Características de mentoría
    mentor_sessions_count: int = Field(..., description="Número de sesiones con mentor")
    mentor_total_words_exchanged: int = Field(..., description="Total de palabras intercambiadas con mentor")
    mentor_before_task_help: int = Field(..., description="Número de ayudas de mentor antes de tareas")
    mentor_after_low_grade: int = Field(..., description="Número de ayudas de mentor después de calificaciones bajas")

    # Variable objetivo
    risk_level: Optional[int] = Field(None, ge=0, lt=3, description="Nivel de riesgo (0: bajo, 1: medio, 2: alto)")

class TrainModelRequest(BaseModel):
    students_data: List[StudentData] = Field(..., description="Lista de datos de estudiantes para entrenamiento")
    model_version: str = Field(..., description="Versión del modelo a entrenar")
    training_date: datetime = Field(default_factory=datetime.utcnow, description="Fecha de entrenamiento")
    description: Optional[str] = Field(None, description="Descripción opcional del entrenamiento")