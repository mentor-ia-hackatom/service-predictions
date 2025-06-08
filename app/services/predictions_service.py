import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import Request, HTTPException
from typing import Dict, List, Tuple
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from app.models.predictions_model import Prediction
from app.schemas.train_model_schema import TrainModelRequest, StudentData
import uuid
import logging
import pickle
import os
from datetime import datetime
import re
import json


logger = logging.getLogger('predictions_service')    

class DBSessionMixin:
    def __init__(self, session: Session):
        self.session = session

class AppDataAccess(DBSessionMixin):
    pass

class AppService(DBSessionMixin):
    def __init__(self, session: Session, request: Request):
        super().__init__(session)
        self.request = request

class PredictionDataAccess(AppDataAccess):
    def save_prediction(self, user_uuid: uuid.UUID, prediction_result: Dict, course_id:str) -> Prediction:
        prediction = Prediction(
            user_uuid=user_uuid,
            prediction_result=prediction_result,
            created_at=int(datetime.now().timestamp()),
            course_id=course_id
        )
        self.session.add(prediction)
        self.session.flush()
        self.session.refresh(prediction)
        return prediction

    def get_last_prediction(self, user_uuid: str, course_id: str, course_code: str) -> Dict[str, float]:
        try:
            item = self.session.query(Prediction)
            item = item.filter(Prediction.user_uuid == user_uuid)
            if course_id:
                item = item.filter(Prediction.course_id.like(f"%{course_id}%"))
            if course_code:
                item = item.filter(Prediction.course_id.like(f"%{course_code}%"))

            item = item.order_by(Prediction.created_at.desc())
            item = item.first()
            return item.prediction_result if item else None
        except Exception as e:
            logger.error(f"Error al obtener la última predicción: {user_uuid} - {e}")
            raise HTTPException(status_code=500, detail=str(e))

class PredictionService(AppService):
    def __init__(self, session: Session, request: Request):
        super().__init__(session, request)
        self.models = {}
        self.scalers = {}
        self.data_access = PredictionDataAccess(session)
        self.models_dir = "app/model_results"
        
    def _prepare_features(self, df: pd.DataFrame, approach: str) -> Tuple[np.ndarray, List[str]]:
        """Prepara las características según el enfoque específico."""
        if approach == "performance":
            features = [
                'avg_grade', 'grade_stddev', 'retry_rate',
                'avg_delivery_delay_days', 'avg_submission_time_diff_hours'
            ]
        elif approach == "participation":
            features = [
                'total_login_time_hours', 'classes_attended', 'classes_missed',
                'last_login_days_ago', 'resource_interactions'
            ]
        elif approach == "delivery":
            features = [
                'late_submissions_count', 'missing_tasks',
                'completed_tasks', 'total_tasks'
            ]
        elif approach == "mentor":
            features = [
                'mentor_sessions_count', 'mentor_total_words_exchanged',
                'mentor_before_task_help', 'mentor_after_low_grade'
            ]
        else:
            raise ValueError(f"Enfoque no válido: {approach}")
            
        X = df[features].values
        return X, features

    def _save_model(self, model: lgb.Booster, scaler: StandardScaler, approach: str, metrics: Dict):
        """Guarda el modelo y sus métricas en archivos pickle."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"{approach}_model_{timestamp}.pkl")
        scaler_path = os.path.join(self.models_dir, f"{approach}_scaler_{timestamp}.pkl")
        metrics_path = os.path.join(self.models_dir, f"{approach}_metrics_{timestamp}.pkl")
        
        # Guardar modelo
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Guardar scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        # Guardar métricas
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
            
        return model_path

    def train_models(self, data:TrainModelRequest) -> Dict[str, lgb.Booster]:
        try:
            """Entrena los cuatro modelos LightGBM para diferentes enfoques."""
            approaches = ["performance", "participation", "delivery", "mentor"]
            training_results = {}

            # Convertir la lista de estudiantes a DataFrame
            df_data = pd.DataFrame([student.dict() for student in data.students_data])

            for approach in approaches:
                X, features = self._prepare_features(df_data, approach)
                y = df_data['risk_level'].values
                
                # Para conjuntos pequeños, usar una división 80-20 sin estratificación
                if len(df_data) < 10:
                    # Si hay muy pocos datos, usar todos para entrenamiento
                    X_train, X_test, y_train, y_test = X, X, y, y
                else:
                    # Para conjuntos más grandes, usar división normal
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                
                # Escalar características
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Configurar y entrenar modelo con parámetros ajustados para datos pequeños
                params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 5,
                    'learning_rate': 0.05,
                    'feature_fraction': 1.0,
                    'min_data_in_leaf': 2,
                    'min_sum_hessian_in_leaf': 1e-2,
                    'max_depth': 2,
                    'min_gain_to_split': 0.0,
                    'lambda_l1': 0.1,
                    'lambda_l2': 0.1,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'verbose': -1
                }
                
                train_data = lgb.Dataset(X_train_scaled, label=y_train)
                valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=30,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=3),
                        lgb.log_evaluation(period=1)
                    ]
                )
                
                # Calcular métricas
                y_pred = model.predict(X_test_scaled)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_classes),
                    'f1_macro': f1_score(y_test, y_pred_classes, average='macro'),
                    'f1_micro': f1_score(y_test, y_pred_classes, average='micro'),
                    'classification_report': classification_report(y_test, y_pred_classes),
                    'features_importance': dict(zip(features, model.feature_importance())),
                    'best_iteration': model.best_iteration,
                    'best_score': model.best_score
                }
                
                # Guardar modelo y métricas
                model_path = self._save_model(model, scaler, approach, metrics)
                
                self.models[approach] = model
                self.scalers[approach] = scaler
                training_results[approach] = {
                    'model_path': model_path,
                    'metrics': metrics
                }
                
            return training_results
        except Exception as e:
            logger.error(f"Error al entrenar los modelos: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _load_latest_model(self, approach: str) -> Tuple[lgb.Booster, StandardScaler]:
        """Carga el modelo y scaler más recientes para un enfoque específico."""
        try:
            # Buscar archivos del modelo más reciente
            model_files = [f for f in os.listdir(self.models_dir) if f.startswith(f"{approach}_model_")]
            if not model_files:
                raise FileNotFoundError(f"No se encontraron modelos para el enfoque {approach}")
            
            # Ordenar por timestamp y obtener el más reciente
            latest_model = sorted(model_files)[-1]
            timestamp = re.search(r'(\d{8}_\d{6})', latest_model).group(1)
            
            # Cargar modelo y scaler
            model_path = os.path.join(self.models_dir, latest_model)
            scaler_path = os.path.join(self.models_dir, f"{approach}_scaler_{timestamp}.pkl")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo {approach}: {e}")
            raise HTTPException(status_code=500, detail=f"Error al cargar el modelo {approach}: {str(e)}")

    def _get_risk_label(self, risk_percentage: float) -> str:
        """Convierte un porcentaje de riesgo en una etiqueta descriptiva."""
        if risk_percentage < 33:
            return "Low"
        elif risk_percentage < 66:
            return "Medium"
        else:
            return "High"

    def process_student_data(self, student_data: StudentData, user_uuid: uuid.UUID, course_id:str) -> Dict[str, float]:
        try:
            """Realiza predicciones de riesgo usando los cuatro modelos y guarda el resultado."""
            predictions = {}
            approaches = ["performance", "participation", "delivery", "mentor"]
            df_data = pd.DataFrame([student_data.dict()])
            
            for approach in approaches:
                try:
                    # Cargar el modelo más reciente
                    model, scaler = self._load_latest_model(approach)
                    
                    # Preparar y escalar características
                    X, _ = self._prepare_features(df_data, approach)
                    X_scaled = scaler.transform(X)
                    
                    # Obtener probabilidades para cada clase
                    probs = model.predict(X_scaled)
                    
                    # Calcular el riesgo promedio (ponderado por las probabilidades)
                    risk_score = np.sum(probs * np.array([0, 1, 2])) / np.sum(probs)
                    
                    # Convertir a porcentaje (0-100)
                    risk_percentage = (risk_score / 2) * 100
                    
                    # Crear diccionario con información detallada
                    predictions[approach] = {
                        "original_value": float(risk_score),
                        "percentage": round(risk_percentage, 2),
                        "risk_level": self._get_risk_label(risk_percentage),
                        "probabilities": {
                            "low_risk": round(float(probs[0][0]) * 100, 2),
                            "medium_risk": round(float(probs[0][1]) * 100, 2),
                            "high_risk": round(float(probs[0][2]) * 100, 2)
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error al procesar {approach}: {e}")
                    predictions[approach] = None
            
            # Calcular riesgo total
            valid_predictions = [v for v in predictions.values() if v is not None]
            if valid_predictions:
                total_percentage = sum(p["percentage"] for p in valid_predictions) / len(valid_predictions)
                predictions['total_risk'] = {
                    "percentage": round(total_percentage, 2),
                    "risk_level": self._get_risk_label(total_percentage),
                    "approach_details": {
                        approach: data["risk_level"] 
                        for approach, data in predictions.items() 
                        if approach != "total_risk" and data is not None
                    }
                }
            else:
                predictions['total_risk'] = None
                logger.error("No se pudo calcular el riesgo total: ninguna predicción válida")
            
            # Guardar la predicción en la base de datos
            saved_prediction = self.data_access.save_prediction(user_uuid, predictions, course_id)
            
            self.session.commit()
            return predictions
        except Exception as e:
            logger.error(f"Error al procesar los datos del estudiante: {user_uuid} - {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    def get_last_prediction(self, user_uuid: str, course_id:str = None, course_code:str = None) -> Dict[str, float]:
        try:
            return self.data_access.get_last_prediction(user_uuid, course_id, course_code)
        except Exception as e:
            logger.error(f"Error al obtener la última predicción: {user_uuid} - {e}")
            raise HTTPException(status_code=500, detail=str(e))