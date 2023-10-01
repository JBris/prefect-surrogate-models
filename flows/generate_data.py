#!/usr/bin/env python

######################################
# Imports
######################################

# External
import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import uuid

# Internal
from .DataModels import CropModel, DirModel, ExperimentModel, SamplingModel
from .sampling import (
    get_inputs,
    get_param_provider,
    sample_params,
    set_override,
    run_till_terminate,
)

######################################
# Functions
######################################

@task
def generate_inputs(
    crop_model: CropModel, experiment_model: ExperimentModel, 
    sampling_model: SamplingModel, dir_model: DirModel
) -> list[dict]:
    """
    Generate input data.

    Args:
        crop_model (CropModel):
            The crop data model.
        experiment_model (ExperimentModel):
            The experiment data model.
        sampling_model (SamplingModel): 
            The sampling data model.
        dir_model (DirModel): 
            The directory data model.

    Returns:
        list[dict]: 
            The input data list.
    """
    agro, weather_data, crop_data, soil_data, site_data = get_inputs(
        dir_model.data_dir,
        dir_model.agro_dir,
        dir_model.crop_dir,
        dir_model.weather_dir,
        crop_model.agro,
        crop_model.latitude,
        crop_model.longitude,
        crop_model.WAV
    )
        
    params = get_param_provider(agro, crop_data, soil_data, site_data)

    sample_scaled = sample_params(
        sampling_model.params, sampling_model.n, experiment_model.seed
    ) 
    columns = list(sampling_model.params.keys())

    df = pd.DataFrame(sample_scaled, columns=columns)
    uuids = [str(uuid.uuid4()) for _ in range(sampling_model.n)]
    df["uuid"] = uuids

    outfile = Path(dir_model.data_dir, dir_model.simulation_dir, "simulation_parameters.csv")
    df.to_csv(str(outfile), index=False)

    params_list = df.to_dict(orient="records")
    return params_list, agro, weather_data, params

@task
def generate_outputs(
    params_list: list[dict], agro: dict, weather_data: NASAPowerWeatherDataProvider, 
    params: ParameterProvider
) -> list[dict]:
    """
    Generate output data.

    Args:
        params_list (list[dict]): 
            The parameter list.
        agro (dict): 
            The agronomy configuration.
        weather_data (NASAPowerWeatherDataProvider): 
            The weather data provider.
        params (ParameterProvider): 
            The parameter provider.

    Returns:
        list[dict]: 
            The simulation output data.
    """
    target_results = []

    for param_set in params_list:
        id = param_set["uuid"]
        del param_set["uuid"]
        params = set_override(params, param_set)
        sim_results = run_till_terminate(params, weather_data, agro)
        sim_results["uuid"] = id
        target_results.append(sim_results)

    return target_results

@flow(
    name="Generate Data",
    description="Generate input and output data for the simulation.",
    task_runner=SequentialTaskRunner(),
)
def generate_data_flow(
    dir_model: DirModel, crop_model: CropModel, experiment_model: ExperimentModel,
    sampling_model: SamplingModel
):
    """
    The data generation flow.

    Args:
        dir_model (DirModel): 
            The directory data model.
        crop_model (CropModel): 
            The crop data model.
        experiment_model (ExperimentModel): 
            The experiment data model.
        sampling_model (SamplingModel): 
            The sampling data model.
    """

    params_list, agro, weather_data, params = generate_inputs(crop_model, experiment_model, sampling_model, dir_model)
    target_results = generate_outputs(params_list, agro, weather_data, params)

    outfile = Path(dir_model.data_dir, dir_model.simulation_dir, "simulation_results.csv")
    pd.DataFrame(target_results).to_csv(str(outfile), index=False)


######################################
# Main
######################################

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    """
    Generate input and output data.

    Args:
        config (DictConfig):    
            The main configuration.
    """
    CROP_CONFIG = config["crop"]
    DIR_CONFIG = config["dir"]
    EXPERIMENT_CONFIG = config["experiment"]
    SAMPLING_CONFIG = config["sampling"]

    dir_model = DirModel(**DIR_CONFIG)
    crop_model = CropModel(**CROP_CONFIG)
    experiment_model = ExperimentModel(**EXPERIMENT_CONFIG)
    sampling_model = SamplingModel(**SAMPLING_CONFIG)
    generate_data_flow(dir_model, crop_model, experiment_model, sampling_model)

if __name__ == "__main__":
    main()
