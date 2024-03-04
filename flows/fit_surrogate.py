#!/usr/bin/env python

######################################
# Imports
######################################

# External

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import torch
import gpytorch
from mlflow.models import infer_signature

# Internal
from .DataModels import DataModel, SurrogateModel, ExperimentModel
from .data import load_data, to_tensor
from .experiment import begin_experiment
from .surrogate import MultitaskVariationalGPModel

######################################
# Functions
######################################

@task
def fit_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    outputs: list[str],
    device: torch.cuda.device,
    num_latents: int,
    num_epochs: int,
    lr: float,
    experiment_prefix: str,
    tracking_uri: str
) -> tuple[
    MultitaskVariationalGPModel, 
    gpytorch.likelihoods.MultitaskGaussianLikelihood,
    torch.Tensor
]:
    """
    Train a Gaussian process model.

    Args:
        X (pd.DataFrame):
            The input matrix.
        Y (pd.DataFrame):
            The output matrix.
        outputs (list[str]):
            A list of output names.
        device (torch.cuda.device):
            The tensor and model device.
        num_latents (int):
            The number of latent variables.
        num_epochs (int):
            The number of training epochs.
        lr (float):
            The training learning rate.
        experiment_prefix (str):
            The prefix for the experiment name.
        tracking_uri (str):
            The experiment tracking URI.

    Returns:
        tuple[MultitaskVariationalGPModel, gpytorch.likelihoods.MultitaskGaussianLikelihood]:
            The trained model and likelihood.
    """
    begin_experiment("fit_model", experiment_prefix, tracking_uri)
    
    mlflow.log_param("num_latents", num_latents)
    model = MultitaskVariationalGPModel(
        n_col=X.shape[-1], num_latents=num_latents, num_tasks=Y.shape[-1]
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y.shape[-1])

    model.train().to(device)
    likelihood.train().to(device)

    mlflow.log_param("lr", lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Y.size(0))

    mlflow.log_param("num_epochs", num_epochs)
    epoch_interval = int(num_epochs / 10)
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y)
        loss.backward()
        optimizer.step()

        if i % epoch_interval == 0:
            print(f"Epochs: {i}  Loss: {loss.item()}")
            mlflow.log_metric("loss", loss.item(), step=i)

    print(f"Final Loss: {loss.item()}")
    
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(X))
        mean = predictions.mean.cpu().numpy()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.cpu().numpy(), upper.cpu().numpy()

    mslls = gpytorch.metrics.mean_standardized_log_loss(predictions, Y).cpu().detach().numpy() 
    mses = gpytorch.metrics.mean_squared_error(predictions, Y).cpu().numpy()
    maes = gpytorch.metrics.mean_absolute_error(predictions, Y).cpu().numpy()
    coverage_errors = gpytorch.metrics.quantile_coverage_error(predictions, Y).cpu().numpy()

    for i, col in enumerate(outputs):
        mlflow.log_metric(f"msll_{col}",  mslls[i])
        mlflow.log_metric(f"mse_{col}",  mses[i])
        mlflow.log_metric(f"mae_{col}",  maes[i])
        mlflow.log_metric(f"empirical_coverage_rate_{col}",  coverage_errors[i])

    for row, col in np.ndindex(Y.shape):
        if row % int(Y.shape[0] * 0.05) != 0:
            continue
        
        mlflow.log_metric(f"Actual {outputs[col]}", Y[row, col], step = row)
        mlflow.log_metric(f"Predicted mean {outputs[col]}", mean[row, col], step = row)
        mlflow.log_metric(f"Predicted lower {outputs[col]}", lower[row, col], step = row)
        mlflow.log_metric(f"Predicted upper {outputs[col]}", upper[row, col], step = row)

    return model, likelihood

@task
def log_model(
    name: str, model: MultitaskVariationalGPModel, 
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood, 
    parameters_df: pd.DataFrame, outputs_df: pd.DataFrame,
) -> None:
    """
    Log the surrogate model to the registry.

    Args:
        name (str): 
            The model name.
        model (MultitaskVariationalGPModel): 
            The surrogate model.
        likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood): 
            The surrogate model likelihood.
        parameters_df (pd.DataFrame): 
            The parameter dataframe.
        outputs_df (pd.DataFrame): 
            The outputs dataframe.
    """
    signature = infer_signature(
        parameters_df, 
        outputs_df
    )
    
    run_id = mlflow.active_run().info.run_id

    def log_torch_model(model, name):
        mlflow.pytorch.log_model(
            model, artifact_path = name, signature = signature
        )
        model_uri = f"runs:/{run_id}/{name}" 
        mlflow.register_model(model_uri, name)

    log_torch_model(model, name)    
    likelihood_name = f"{name}_likelihood"
    log_torch_model(likelihood, likelihood_name)

    mlflow.end_run()


@flow(
    name="Fit Surrogate",
    description="Fit a surrogate model.",
    task_runner=SequentialTaskRunner(),
)
def fit_surrogate_flow(
    data_model: DataModel, surrogate_model: SurrogateModel,  
    experiment_model: ExperimentModel
) -> None:
    """
    Fitting the surrogate model flow.

    Args:
        data_model (DataModel):
            The input data data model.
        surrogate_model (SurrogateModel):
            The Gaussian process data model.
        experiment_model (OptimisationModel):
            The experiment data model.
    """
    parameters_df, outputs_df, df = load_data(
        data_model.data_dir,
        data_model.input_dir,
        data_model.parameters_file,
        data_model.outputs_file,
        data_model.inputs,
        data_model.outputs,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X, Y, scaler = to_tensor(parameters_df, outputs_df, device)

    model, likelihood = fit_model(
        X, Y, outputs_df.columns, device, surrogate_model.num_latents, surrogate_model.num_epochs, 
        surrogate_model.lr, experiment_model.name, experiment_model.tracking_uri
    )

    log_model(
        surrogate_model.name, model, likelihood, parameters_df, outputs_df
    )

######################################
# Main
######################################


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    """
    Fit a surrogate model.

    Args:
        config (DictConfig):
            The main configuration.
    """
    config = OmegaConf.to_container(config)
    DATA_CONFIG = config["data"]
    EXPERIMENT_CONFIG = config["experiment"]
    SURROGATE_CONFIG = config["surrogate"]

    data_model = DataModel(**DATA_CONFIG)
    experiment_model = ExperimentModel(**EXPERIMENT_CONFIG)
    surrogate_model = SurrogateModel(**SURROGATE_CONFIG)

    fit_surrogate_flow(
        data_model, surrogate_model, experiment_model
    )

if __name__ == "__main__":
    main()