from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Request, Depends, HTTPException
from sqlalchemy.orm import Session
from app.utils.dataBase import get_db
from app.services.predictions_service import PredictionService
from app.schemas.train_model_schema import TrainModelRequest, StudentData

router = APIRouter(prefix="/internal/process/predictions", tags=["Predictions"])

@router.post("/process_student_data")
async def process_student_data(
    student_data: StudentData, 
    user_uuid: str, 
    course_id: str,
    request: Request, 
    background_tasks: BackgroundTasks, 
    session: Session = Depends(get_db)
):

    try: 
        background_tasks.add_task(
            PredictionService(session, request).process_student_data,
            student_data=student_data,
            user_uuid=user_uuid,
            course_id=course_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "status": "ok",
        "version": "1.0.0",
        "message": "Prediction task started in background"
    }
    
@router.post("/train_models")
async def train_models(
    data: TrainModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db)
):
    try:
        background_tasks.add_task(
            PredictionService(session, request).train_models,
            data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "status": "ok",
        "version": "1.0.0",
        "message": "Training task started in background"
    }

@router.get("/get_last_prediction")
async def get_last_prediction(
    request: Request,
    user_uuid: str,
    course_id: Optional[str] = None,
    course_code: Optional[str] = None,
    session: Session = Depends(get_db)
):
    try:
        return PredictionService(session, request).get_last_prediction(user_uuid, course_id, course_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))