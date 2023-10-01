#!/usr/bin/env python

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from DataModels import CropModel

import hydra
from omegaconf import DictConfig

import pandas as pd

import pandas as pd

from sampling import (
    get_inputs,
    get_param_provider,
    sample_params,
    set_override,
    run_till_terminate,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_flow(config: DictConfig):
    CROP_CONFIG = config["crop"]
    DIR_CONFIG = config["dir"]
    EXPERIMENT_CONFIG = config["experiment"]
    SAMPLING_CONFIG = config["sampling"]

    agro, weather_data, crop_data, soil_data, site_data = get_inputs(
        DIR_CONFIG["data_dir"],
        DIR_CONFIG["agro_dir"],
        DIR_CONFIG["crop_dir"],
        DIR_CONFIG["weather_dir"],
        CROP_CONFIG["agro"],
        CROP_CONFIG["latitude"],
        CROP_CONFIG["longitude"],
        CROP_CONFIG["WAV"],
    )

    params = get_param_provider(agro, crop_data, soil_data, site_data)
    sample_scaled = sample_params(
        SAMPLING_CONFIG["params"], SAMPLING_CONFIG["n"], EXPERIMENT_CONFIG["seed"]
    )

    import uuid

    uuids = [str(uuid.uuid4()) for _ in range(SAMPLING_CONFIG["n"])]
    columns = list(SAMPLING_CONFIG["params"].keys())

    df = pd.DataFrame(sample_scaled, columns=columns)
    df["uuid"] = uuids
    df.to_csv("params.csv", index=False)
    params_list = df.to_dict(orient="records")
    target_results = []
    for param_set in params_list:
        id = param_set["uuid"]
        del param_set["uuid"]
        params = set_override(params, param_set)
        sim_results = run_till_terminate(params, weather_data, agro)
        sim_results["uuid"] = id
        target_results.append(sim_results)

    pd.DataFrame(target_results).to_csv("out.csv", index=False)

if __name__ == "__main__":
    main_flow()
