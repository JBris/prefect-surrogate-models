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
    processed_dir: str
    
class DataModel(BaseModel):
    """
    The data model.

    Args:
        BaseModel (_type_):
            The Base model class.
    """

    data_dir: str
    input_dir: str
    output_dir: str
    parameters_file: str
    outputs_file: str
    parameters_test_file: str
    outputs_test_file: str
    inputs: list[str]
    outputs: list[str]
    
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
    n_workers: int
    threads_per_worker: int
    parameters_file: str
    outputs_file: str
    test_row: int
    
class SamplingModel(BaseModel):
    """
    The sampling data model.

    Args:
        BaseModel (_type_): 
            The Base model class.
    """
    params: dict
    n: int

class SurrogateModel(BaseModel):
    """
    The surrogate model.

    Args:
        BaseModel (_type_):
            The Base model class.
    """
    name: str
    num_latents: int
    num_epochs: int
    lr: float


class OptimisationModel(BaseModel):
    """
    The optimisation model.

    Args:
        BaseModel (_type_):
            The Base model class.
    """

    n_trials: int
    n_jobs: int
    params: dict