from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Date, ForeignKey, Text, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///database.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Person(Base):
    __tablename__ = "persons"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(100), nullable=False)
    employee_id = Column(String(50), unique=True, nullable=False)
    role        = Column(String(100), default="Employee")
    department  = Column(String(100), default="General")
    photo_path  = Column(String(255), nullable=True)      # saved face snapshot
    created_at  = Column(DateTime, default=datetime.utcnow)
    is_active   = Column(Boolean, default=True)

    embeddings  = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")
    logs        = relationship("AttendanceLog",  back_populates="person", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":          self.id,
            "name":        self.name,
            "employee_id": self.employee_id,
            "role":        self.role,
            "department":  self.department,
            "photo_path":  self.photo_path,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
            "is_active":   self.is_active,
        }


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    person_id   = Column(Integer, ForeignKey("persons.id"), nullable=False)
    embedding   = Column(Text, nullable=False)   # JSON-serialised float list
    model_name  = Column(String(50), default="Facenet512")
    created_at  = Column(DateTime, default=datetime.utcnow)

    person      = relationship("Person", back_populates="embeddings")


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    person_id    = Column(Integer, ForeignKey("persons.id"), nullable=False)
    date         = Column(Date, default=date.today)
    check_in     = Column(DateTime, nullable=True)
    check_out    = Column(DateTime, nullable=True)
    emotion      = Column(String(50), nullable=True)
    confidence   = Column(Float, default=0.0)
    is_present   = Column(Boolean, default=True)

    person       = relationship("Person", back_populates="logs")

    def to_dict(self):
        return {
            "id":          self.id,
            "person_id":   self.person_id,
            "person_name": self.person.name if self.person else "Unknown",
            "employee_id": self.person.employee_id if self.person else "",
            "department":  self.person.department if self.person else "",
            "date":        self.date.isoformat() if self.date else None,
            "check_in":    self.check_in.isoformat() if self.check_in else None,
            "check_out":   self.check_out.isoformat() if self.check_out else None,
            "emotion":     self.emotion,
            "confidence":  round(self.confidence, 3),
            "is_present":  self.is_present,
        }


def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()
