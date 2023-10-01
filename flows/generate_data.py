#!/usr/bin/env python

######################################
# Imports
######################################

# External
import hydra
import mlflow
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from prefect import flow, task
from prefect.artifacts import create_table_artifact
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

    outfile = Path(dir_model.data_dir, dir_model.simulation_dir, "parameters.csv")
    df.to_csv(str(outfile), index=False)

    params_list = df.to_dict(orient="records")
    return params_list, agro, weather_data, params, outfile

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
) -> tuple[str, str]:
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

    Returns:
        tuple[str, str]: 
            The parameter and output file paths.
    """

    params_list, agro, weather_data, params, params_file = generate_inputs(
        crop_model, experiment_model, sampling_model, dir_model
    )
    target_results = generate_outputs(params_list, agro, weather_data, params)

    outfile = Path(dir_model.data_dir, dir_model.simulation_dir, "outputs.csv")
    out_df = pd.DataFrame(target_results)
    out_df.to_csv(str(outfile), index=False)

    create_table_artifact(
        key="crop-input-data",
        table=params_list,
        description= "Crop simulation input data."
    )

    return params_file, outfile

def log_results(
    tracking_uri: str, experiment_prefix: str,
    crop_config: dict, dir_config: dict, 
    experiment_config: dict, sampling_config: dict,
    params_file: str, output_file: str
) -> None: 
    """
    Log experiment results to the experiment tracker.

    Args:
        tracking_uri (str):
            The tracking URI.
        experiment_prefix (str):
            The experiment name prefix.
        crop_config (dict):
            The crop configuration.
        dir_config (dict):
            The directory configuration.
        experiment_config (dict):
            The experiment configuration.
        sampling_config (dict):
            The sampling configuration.
        params_file (str):
            The parameters file path.
        output_file (str):
            The output file path.
    """
    task = "generate_data"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{experiment_prefix}_{task}"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.set_tag("task", task)
    
    for config in [crop_config, dir_config, experiment_config, sampling_config]:
        for k in config:
            mlflow.log_param(k, config[k])
        
    mlflow.log_artifact(params_file)
    mlflow.log_artifact(output_file)
    mlflow.end_run()

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
    params_file, output_file = generate_data_flow(
        dir_model, crop_model, experiment_model, sampling_model
    )

    if experiment_model.tracking_enabled:
        log_results(
            experiment_model.tracking_uri, experiment_model.name, CROP_CONFIG, 
            DIR_CONFIG, EXPERIMENT_CONFIG, SAMPLING_CONFIG, params_file, output_file
        )
        
if __name__ == "__main__":
    main()
