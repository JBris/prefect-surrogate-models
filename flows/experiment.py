######################################
# Imports
######################################

import mlflow
from pathlib import Path
import sys

######################################
# Functions
######################################

def get_experiment_name() -> str:
    """
    Get the experiment name.

    Returns:
        str: The experiment name.
    """
    return Path(
        sys.argv[0]
    ).stem

def begin_experiment(
    task: str, experiment_prefix: str, tracking_uri: str
):
    """
    Begin the experiment session.

    Args:   
        task (str):
            The name of the current task for the experiment.
        experiment_prefix (str):
            The prefix for the experiment name.
        tracking_uri (str):
            The experiment tracking URI.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{experiment_prefix}_{task}"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.set_tag("task", task)
