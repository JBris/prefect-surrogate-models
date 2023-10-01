#!/usr/bin/env python

######################################
# Imports
######################################

# External
import hydra
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
    train_df = df.query("uuid != @test_uuid")
    test_df = df.query("uuid == @test_uuid")

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
) -> None:
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
            The outputs dataframe
    """
    Path(data_dir, processed_dir).mkdir(parents=True, exist_ok=True)
    save_processed_df(
        data_dir, processed_dir, test_uuid, parameters_file, 
        parameters_df
    )
    save_processed_df(
        data_dir, processed_dir, test_uuid, outputs_file, 
        outputs_df
    )

@flow(
    name="Process Data",
    description="Create a train-test split for processed data.",
    task_runner=SequentialTaskRunner(),
)
def process_data_flow(dir_model: DirModel, experiment_model: ExperimentModel) -> None:
    """
    The data processing flow.

    Args:
        dir_model (DirModel):   
            The directory data model.
        experiment_model (ExperimentModel):     
            The experiment data model.
    """
    parameters_df, outputs_df, test_uuid = load_data(
        dir_model.data_dir, dir_model.simulation_dir, experiment_model.parameters_file,
        experiment_model.outputs_file
    )

    save_processed_dfs(
        dir_model.data_dir, dir_model.processed_dir, experiment_model.parameters_file,
        experiment_model.outputs_file, test_uuid, parameters_df, outputs_df
    )

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

    process_data_flow(dir_model, experiment_model)

if __name__ == "__main__":
    main()
