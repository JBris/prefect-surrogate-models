######################################
# Imports
######################################

import pandas as pd
from pathlib import Path
from prefect import task
from sklearn.preprocessing import MinMaxScaler
import torch

######################################
# Functions
######################################

@task
def load_data(
    data_dir: str,
    input_dir: str,
    parameters_file: str,
    outputs_file: str,
    inputs: list[str],
    outputs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the parameters and outputs dataframes.

    Args:
        data_dir (str):
            The data directory.
        input_dir (str):
            The data input directory.
        parameters_file (str):
            The parameters dataframe filename.
        outputs_file (str):
            The outputs dataframe filename.
        inputs (list[str]):
            The list of input columns.
        outputs (list[str]):
            The list of output columns.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            The parameters and outputs dataframes.
    """
    parameters_file = Path(data_dir, input_dir, parameters_file)
    outputs_file = Path(data_dir, input_dir, outputs_file)

    parameters_df = pd.read_csv(str(parameters_file))
    outputs_df = pd.read_csv(str(outputs_file))
    df = pd.merge(parameters_df, outputs_df, on = "uuid")

    parameters_df = parameters_df[inputs]
    outputs_df = outputs_df[outputs]
    df = df[inputs + outputs]

    return parameters_df, outputs_df, df

@task
def to_tensor(
    parameters_df: pd.DataFrame, outputs_df: pd.DataFrame, 
    device: torch.cuda.device, scaler: MinMaxScaler = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dataframes to tensors.

    Args:
        parameters_df (pd.DataFrame):
            The parameter dataframe.
        outputs_df (pd.DataFrame):
            The output dataframe.
        device (torch.cuda.device):
            The tensor device.
        scaler (MinMaxScaler):
            The data scaler.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            The parameter and output tensors.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(parameters_df)

    X = torch.from_numpy(scaler.transform(parameters_df)).float().to(device)
    Y = torch.from_numpy(outputs_df.values).float().to(device)

    return X, Y, scaler