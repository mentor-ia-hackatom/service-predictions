from fastapi import APIRouter
from app.controller import health
from app.controller import prediction_controller

api_router = APIRouter()
api_router.include_router(health.router) 
api_router.include_router(prediction_controller.router) 