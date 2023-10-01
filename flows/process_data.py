#!/usr/bin/env python

######################################
# Imports
######################################

# External
import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path

# Internal
from .DataModels import CropModel, DirModel, ExperimentModel, SamplingModel

######################################
# Functions
######################################

def save_processed_df(
    data_dir: str, processed_dir: str, test_uuid: str, filename: str, df: pd.DataFrame
) -> tuple[str, str]:
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

######################################
# Main
######################################

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    DIR_CONFIG = config["dir"]
    EXPERIMENT_CONFIG = config["experiment"]

    dir_model = DirModel(**DIR_CONFIG)
    experiment_model = ExperimentModel(**EXPERIMENT_CONFIG)

    parameters_file = Path(
        dir_model.data_dir, dir_model.simulation_dir, experiment_model.parameters_file
    )
    outputs_file = Path(
        dir_model.data_dir, dir_model.simulation_dir, experiment_model.outputs_file
    )
    
    parameters_df = pd.read_csv(str(parameters_file))
    outputs_df = pd.read_csv(str(outputs_file))

    test_uuid = parameters_df.uuid.iloc[0]
    Path(dir_model.data_dir, dir_model.processed_dir).mkdir(parents=True, exist_ok=True)
    
    save_processed_df(
        dir_model.data_dir, dir_model.processed_dir, test_uuid, experiment_model.parameters_file, 
        parameters_df
    )
    save_processed_df(
        dir_model.data_dir, dir_model.processed_dir, test_uuid, experiment_model.outputs_file, 
        outputs_df
    )

if __name__ == "__main__":
    main()
