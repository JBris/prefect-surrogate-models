from pydantic import BaseModel

class DirModel(BaseModel):
    """
    The directory data model.

    Args:
        BaseModel (_type_): 
            The Base model class.
    """
    data_dir: str
    agro_dir: str
    crop_dir: str
    weather_dir: str
    simulation_dir: str
    out_dir: str
    
class CropModel(BaseModel):
    """
    The crop data model.

    Args:
        BaseModel (_type_): 
            The Base model class.
    """
    latitude: float
    longitude: float
    WAV: float
    agro: str

class ExperimentModel(BaseModel):
    """
    The experiment data model.

    Args:
        BaseModel (_type_): 
            The Base model class.
    """
    name: str
    tracking_uri: str
    tracking_enabled: bool
    seed: int

class SamplingModel(BaseModel):
    """
    The sampling data model.

    Args:
        BaseModel (_type_): 
            The Base model class.
    """
    params: dict
    n: int
