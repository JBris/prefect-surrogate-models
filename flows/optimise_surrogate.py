#!/usr/bin/env python

######################################
# Imports
######################################

# External

import hydra
from joblib import dump as optimiser_dump 
import kaleido
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from pathlib import Path
from prefect.artifacts import create_table_artifact
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.preprocessing import MinMaxScaler
import torch
import gpytorch
from ydata_profiling import ProfileReport

# Internal
from .DataModels import DataModel, SurrogateModel, ExperimentModel, OptimisationModel
from .data import load_data, to_tensor
from .experiment import get_experiment_name, begin_experiment
from .surrogate import load_surrogate, MultitaskVariationalGPModel

######################################
# Functions
######################################

def objective(
    trial,
    model: MultitaskVariationalGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    scaler: MinMaxScaler,
    parameters_df_bounds: pd.DataFrame,
    device: torch.cuda.device,
    outputs_df_test: pd.DataFrame
) -> tuple[any]:
    """
    The optimisation objective function.

    Args:
        trial (_type_):
            The trial object.
        model (MultitaskVariationalGPModel):
            The surrogate model.
        likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood):
            The Gaussian process likelihood.
        scaler (MinMaxScaler):
            The data scaler.
        parameters_df_bounds (pd.DataFrame):
            The bounds of the input parameter dataframe.
        device (torch.cuda.device):
            The tensor device.

    Returns:
        tuple[any]: 
            The predicted values.
    """
    sample = {}
    for col in parameters_df_bounds.columns:
        min_val = parameters_df_bounds.iloc[0][col]
        max_val = parameters_df_bounds.iloc[-1][col]
        sample[col] = trial.suggest_float(col, min_val, max_val)

    samples_scaled = scaler.transform(pd.DataFrame.from_records([sample]))
    samples_tensor = torch.from_numpy(samples_scaled).float().to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(samples_tensor))
        mean = predictions.mean.cpu().numpy().reshape(-1) 

    outputs = outputs_df_test.values.reshape(-1) 
    discrepency = np.linalg.norm(mean - outputs, ord = 2)
    return discrepency

@task
def optimise_model(
    model: MultitaskVariationalGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    scaler: MinMaxScaler,
    parameters_df: pd.DataFrame,
    parameters_df_test: pd.DataFrame,
    outputs_df_test: pd.DataFrame,
    n_trials: int,
    n_jobs: int,
    params: dict,
    device: torch.cuda.device,
    data_dir: str,
    output_dir: str,
    experiment_prefix: str,
    tracking_uri: str,
    df: pd.DataFrame
) -> None:
    """
    Optimise the surrogate model.

    Args:
        model (MultitaskVariationalGPModel):
            The surrogate model.
        likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood):
            The Gaussian process likelihood.
        scaler (MinMaxScaler):
            The data scaler.
        parameters_df (pd.DataFrame):
            The input parameter dataframe.
        parameters_df_test (pd.DataFrame):
            The ground-truth input parameter dataframe.
        outputs_df_test (pd.DataFrame):
            The ground-truth outputs dataframe.
        n_trials (int):
            The number of optimisation trials.
        n_jobs (int):
            The number of trials to run in parallel.
        params (dict):
            The optimisation sampler parameter dictionary.
        device (torch.cuda.device):
            The tensor device.
        data_dir (str):
            The data directory.
        output_dir (str):
            The data output directory.
        experiment_prefix (str):
            The prefix for the experiment name.
        tracking_uri (str):
            The experiment tracking URI.
        df (pd.DataFrame):
            The combined dataframe.
    """
    begin_experiment(get_experiment_name(), experiment_prefix, tracking_uri)

    parameters_df_bounds = parameters_df.agg(["min", "max"])
    sampler = TPESampler(**params)
    for k in params:
        mlflow.log_param(k, params.get(k))
    
    study = optuna.create_study(sampler=sampler, direction="minimize")

    mlflow.log_param("n_trials", n_trials)
    study.optimize(
        lambda trial: objective(
            trial, model, likelihood, scaler, parameters_df_bounds, device,
            outputs_df_test
        ),
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=(n_jobs == 1),
    )

    outpath = Path(data_dir, output_dir)
    outpath.mkdir(parents=True, exist_ok=True)
    outdir = str(outpath)

    trials_df = study.trials_dataframe()
    trials_out = str(Path(outdir, "optimisation_results.csv"))
    trials_df.to_csv(trials_out, index = False)
    mlflow.log_artifact(trials_out)

    create_table_artifact(
        key="opt-params",
        table=trials_df.drop(
            columns=["datetime_start", "datetime_complete", "duration"]
        ).sort_values(by="value").head(10).to_dict(orient="records"),
        description= "# Optimised parameters for simulation study."
    )

    parameters_test_out = str(Path(outdir, "parameters_test.csv"))
    parameters_df_test.to_csv(parameters_test_out, index = False)
    mlflow.log_artifact(parameters_test_out)

    create_table_artifact(
        key="ground-truth-params",
        table=parameters_df_test.to_dict(orient="records"),
        description= "# Ground truth parameters for simulation study."
    )

    optimiser_file = str(Path(outdir, "optimiser.pkl") )
    optimiser_dump(study, optimiser_file)
    mlflow.log_artifact(optimiser_file)

    for plot_func in [
        optuna.visualization.plot_edf, optuna.visualization.plot_optimization_history,
        optuna.visualization.plot_parallel_coordinate, optuna.visualization.plot_param_importances,
        optuna.visualization.plot_slice
    ]:
        img_file = str(Path(outdir, f"{plot_func.__name__}.png"))
        plot_func(study).write_image(
            img_file
        )
        mlflow.log_artifact(img_file)

    profile = ProfileReport(df, title = experiment_prefix)
    profile_file = str(Path(outdir, f"{experiment_prefix}.html"))
    profile.to_file(profile_file)
    mlflow.log_artifact(profile_file)

    mlflow.end_run()

@flow(
    name="Optimise Surrogate",
    description="Perform surrogate optimisation.",
    task_runner=SequentialTaskRunner(),
)
def optimise_surrogate_flow(
    data_model: DataModel, surrogate_model: SurrogateModel, 
    optimisation_model: OptimisationModel, experiment_model: ExperimentModel
) -> None:
    """
    The optimisation surrogate flow.

    Args:
        data_model (DataModel):
            The input data data model.
        surrogate_model (SurrogateModel):
            The Gaussian process data model.
        optimisation_model (OptimisationModel):
            The optimisation data model.
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
    _, _, scaler = to_tensor(parameters_df, outputs_df, device)

    parameters_df_test, outputs_df_test, _ = load_data(
        data_model.data_dir,
        data_model.input_dir,
        data_model.parameters_test_file,
        data_model.outputs_test_file,
        data_model.inputs,
        data_model.outputs,
    )

    model, likelihood = load_surrogate(surrogate_model.name)

    optimise_model(
        model,
        likelihood,
        scaler,
        parameters_df,
        parameters_df_test,
        outputs_df_test,
        optimisation_model.n_trials,
        optimisation_model.n_jobs,
        optimisation_model.params,
        device,
        data_model.data_dir,
        data_model.output_dir,
        experiment_model.name,
        experiment_model.tracking_uri,
        df
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
    OPTIMISATION_CONFIG = config["optimise"]

    data_model = DataModel(**DATA_CONFIG)
    experiment_model = ExperimentModel(**EXPERIMENT_CONFIG)
    surrogate_model = SurrogateModel(**SURROGATE_CONFIG)
    optimisation_model = OptimisationModel(**OPTIMISATION_CONFIG)

    optimise_surrogate_flow(
        data_model, surrogate_model, optimisation_model, experiment_model
    )

if __name__ == "__main__":
    main()