from pydantic import BaseModel

class CropModel(BaseModel):
    latitude: float
    longitude: float
    WAV: float
