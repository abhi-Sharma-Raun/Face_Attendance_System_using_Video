from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    num_eval_nbd_frames: int
    sharpness_threshold: int
    face_size_threshold: int
    pose_symmetry_threshold: float
    class Config:
        env_file = ".env"
    
settings = Settings()
    