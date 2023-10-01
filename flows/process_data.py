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
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

# Internal
from .DataModels import DirModel, ExperimentModel

######################################
# Functions
######################################

@task
def load_data(
    data_dir: str, simulation_dir: str, parameters_file: str, outputs_file: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load the parameters and outputs dataframes.

    Args:
        data_dir (str): 
            The data directory.
        simulation_dir (str): 
            The simulation directory.
        parameters_file (str): 
            The parameters dataframe filename.
        outputs_file (str):
            The outputs dataframe filename.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, str]: 
            The parameters and outputs dataframes.
    """
    parameters_file = Path(
        data_dir, simulation_dir, parameters_file
    )
    outputs_file = Path(
        data_dir, simulation_dir, outputs_file
    )
    
    parameters_df = pd.read_csv(str(parameters_file))
    outputs_df = pd.read_csv(str(outputs_file))
    test_uuid = parameters_df.uuid.iloc[0]

    return parameters_df, outputs_df, test_uuid

def save_processed_df(
    data_dir: str, processed_dir: str, test_uuid: str, filename: str, df: pd.DataFrame
) -> tuple[str, str]:
    """
    Save the train-test data.

    Args:
        data_dir (str): 
            The data directory.
        processed_dir (str): 
            The processed directory.
        test_uuid (str): 
            The test UUID.
        filename (str): 
            The file name.
        df (pd.DataFrame): 
            The input dataframe.

    Returns:
        tuple[str, str]: 
            The train-test output file names.
    """

    get_df = lambda query, test_uuid: (
        df
        .query(query)
        .select_dtypes(include='number')
        .dropna(axis = 1)
    )
    
    train_df = get_df("uuid != @test_uuid", test_uuid) 
    test_df = get_df("uuid == @test_uuid", test_uuid) 

    get_outfile = lambda prefix : str(Path(
        data_dir, processed_dir, f"{prefix}_{filename}"
    ))

    train_outfile = get_outfile("train")
    train_df.to_csv(train_outfile, index=False)

    test_outfile = get_outfile("test")
    test_df.to_csv(test_outfile, index=False)

    return train_outfile, test_outfile

@task
def save_processed_dfs(
    data_dir: str, processed_dir: str, parameters_file: str, outputs_file: str,
    test_uuid: str, parameters_df: pd.DataFrame, outputs_df: pd.DataFrame
) -> tuple[str, str, str, str]:
    """
    Save the processed dataframes.

    Args:
        data_dir (str): 
            The data directory.
        processed_dir (str): 
            The processed data directory.
        parameters_file (str): 
            The parameters dataframe file.
        outputs_file (str): 
            The outputs dataframe file.
        test_uuid (str): 
            The test UUID value.
        parameters_df (pd.DataFrame): 
            The parameters dataframe.
        outputs_df (pd.DataFrame): 
            The outputs dataframe.

    Returns:
        tuple[str, str, str, str]: 
            The train-test parameter and output file names.
    """
    Path(data_dir, processed_dir).mkdir(parents=True, exist_ok=True)
    train_parameters_file, test_parameters_file = save_processed_df(
        data_dir, processed_dir, test_uuid, parameters_file, 
        parameters_df
    )
    train_outputs_file, test_outputs_file = save_processed_df(
        data_dir, processed_dir, test_uuid, outputs_file, 
        outputs_df
    )

    return train_parameters_file, test_parameters_file, train_outputs_file, test_outputs_file

@flow(
    name="Process Data",
    description="Create a train-test split for processed data.",
    task_runner=SequentialTaskRunner(),
)
def process_data_flow(dir_model: DirModel, experiment_model: ExperimentModel) -> tuple[str, str, str, str]:
    """
    The data processing flow.

    Args:
        dir_model (DirModel):   
            The directory data model.
        experiment_model (ExperimentModel):     
            The experiment data model.

    Returns:
        tuple[str, str, str, str]: 
            The train-test parameter and output file names.
    """
    parameters_df, outputs_df, test_uuid = load_data(
        dir_model.data_dir, dir_model.simulation_dir, experiment_model.parameters_file,
        experiment_model.outputs_file
    )

    train_parameters_file, test_parameters_file, train_outputs_file, test_outputs_file = save_processed_dfs(
        dir_model.data_dir, dir_model.processed_dir, experiment_model.parameters_file,
        experiment_model.outputs_file, test_uuid, parameters_df, outputs_df
    )

    return train_parameters_file, test_parameters_file, train_outputs_file, test_outputs_file

def log_results(
    tracking_uri: str, experiment_prefix: str, dir_config: dict, experiment_config: dict, 
    train_parameters_file: str, test_parameters_file: str, train_outputs_file: str, 
    test_outputs_file: str 
) -> None:
    """
    Log experiment results to the experiment tracker.

    Args:
        tracking_uri (str):
            The tracking URI.
        experiment_prefix (str):
            The experiment name prefix.
        dir_config (dict):
            The directory configuration.
        experiment_config (dict):
            The experiment configuration.
        train_parameters_file (str):
            The training data for simulation parameters.
        test_parameters_file: (str):
            The testing data for simulation parameters.
        train_outputs_file: (str):  
            The training data for simulation outputs.
        test_outputs_file: (str): 
            The testing data for simulation outputs.
    """
    task = "process_data"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{experiment_prefix}_{task}"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.set_tag("task", task)
    
    for config in [dir_config, experiment_config]:
        for k in config:
            mlflow.log_param(k, config[k])
    
    for f in [train_parameters_file, test_parameters_file, train_outputs_file, test_outputs_file]:
        mlflow.log_artifact(f)

    mlflow.end_run()

######################################
# Main
######################################

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    """
    Process simulation data resulting in a train-test split.

    Args:
        config (DictConfig):    
            The main configuration.
    """
    DIR_CONFIG = config["dir"]
    EXPERIMENT_CONFIG = config["experiment"]

    dir_model = DirModel(**DIR_CONFIG)
    experiment_model = ExperimentModel(**EXPERIMENT_CONFIG)

    train_parameters_file, test_parameters_file, train_outputs_file, test_outputs_file = process_data_flow(
        dir_model, experiment_model
    )

    if experiment_model.tracking_enabled:
        log_results(
            experiment_model.tracking_uri, experiment_model.name, 
            DIR_CONFIG, EXPERIMENT_CONFIG, train_parameters_file, 
            test_parameters_file, train_outputs_file, test_outputs_file 
        )

if __name__ == "__main__":
    main()
