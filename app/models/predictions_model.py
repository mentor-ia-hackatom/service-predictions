from datetime import datetime
import uuid
from sqlalchemy import Column, DateTime, String, JSON, BigInteger
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.utils.dataBase import Base

class Prediction(Base):
    __tablename__ = "predictions"

    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(BigInteger, nullable=False)
    user_uuid = Column(UUID(as_uuid=True), nullable=False)
    course_id = Column(String, nullable=False)
    prediction_result = Column(JSONB, nullable=False)

    def __repr__(self):
        return f"<Prediction(id={self.id}, user_uuid={self.user_uuid}, created_at={self.created_at})>"
