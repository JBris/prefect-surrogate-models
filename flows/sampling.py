#!/usr/bin/env python

######################################
# Imports
######################################

import numpy as np
from os.path import join as join_path
import pcse
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.models import Wofost72_PP
from pcse.util import DummySoilDataProvider, WOFOST72SiteDataProvider
from scipy.stats import qmc
import yaml

######################################
# Functions
######################################


def get_inputs(
    data_dir: str,
    agro_dir: str,
    crop_dir: str,
    weather_dir: str,
    agro_file: str,
    latitude: float,
    longitude: float,
    WAV: float,
) -> tuple[
    dict,
    NASAPowerWeatherDataProvider,
    YAMLCropDataProvider,
    DummySoilDataProvider,
    WOFOST72SiteDataProvider,
]:
    """
    Get input data.

    Args:
        data_dir (str):
            The data directory.
        agro_dir (str):
            The agronomy directory.
        crop_dir (str):
            The crop directory.
        weather_dir (str):
            The weather directory.
        agro_file (str):
            The agronomy config file.
        latitude (float):
            The crop latitude.
        longitude (float):
            The crop longitude.
        WAV (float):

    Returns:
        tuple[
            dict, NASAPowerWeatherDataProvider, YAMLCropDataProvider, DummySoilDataProvider, WOFOST72SiteDataProvider
        ]:
            The input data.
    """
    agro_config = join_path(data_dir, agro_dir, agro_file)
    meteo_dir = join_path(data_dir, weather_dir)
    crop_dir = join_path(data_dir, crop_dir)

    with open(agro_config, encoding="utf8") as f:
        agro = yaml.safe_load(f)

    pcse.settings.METEO_CACHE_DIR = meteo_dir
    weather_data = NASAPowerWeatherDataProvider(latitude=latitude, longitude=longitude)

    crop_data = YAMLCropDataProvider(fpath=crop_dir, force_reload=True)
    soil_data = DummySoilDataProvider()
    site_data = WOFOST72SiteDataProvider(WAV=WAV)

    return agro, weather_data, crop_data, soil_data, site_data


def get_param_provider(
    agro: dict,
    crop_data: YAMLCropDataProvider,
    soil_data: DummySoilDataProvider,
    site_data: WOFOST72SiteDataProvider,
) -> ParameterProvider:
    """
    Get the simulation parameter provider.

    Args:
        agro (dict):
            The agronomy dictionary.
        crop_data (YAMLCropDataProvider):
            The crop data provider.
        soil_data (DummySoilDataProvider):
            The soil data provider.
        site_data (WOFOST72SiteDataProvider):
            The site data provider.

    Returns:
        ParameterProvider:
            The simulation parameter provider.
    """
    first_date = list(agro[0])[0]
    crop_calendar = agro[0][first_date]["CropCalendar"]

    crop_data.set_active_crop(crop_calendar["crop_name"], crop_calendar["variety_name"])
    params = ParameterProvider(
        cropdata=crop_data, sitedata=site_data, soildata=soil_data
    )

    return params


def sample_params(params_config: dict, n: int, seed: int = None) -> np.array:
    """
    Sample from the parameter space using Latin Hypercube sampling.

    Args:
        params_config (dict):
            The parameter configuration containing lower and upper bounds.
        n (int):
            The number of draws.
        seed (int, optional):
            The random seed. Defaults to None.

    Returns:
        np.array:
            Latin Hypercube samples.
    """
    d = len(params_config)
    sampler = qmc.LatinHypercube(d=d, seed=seed)

    lower_bounds = []
    upper_bounds = []
    for col in params_config:
        lb = params_config[col]["lb"]
        ub = params_config[col]["ub"]
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    sample = sampler.random(n=n)

    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    return sample_scaled


def set_override(params: ParameterProvider, param_set: dict) -> ParameterProvider:
    """
    Override parameters for a simulation.

    Args:
        params (ParameterProvider):
            The parameter provider.
        param_set (dict):
            A parameter set dictionary.

    Returns:
        ParameterProvider:
            The parameter provider.
    """
    for k in param_set:
        params.set_override(k, param_set[k])
    return params


def run_till_terminate(
    params: ParameterProvider, weather_data: NASAPowerWeatherDataProvider, agro: dict
) -> dict:
    """
    Run the simulation until termination.

    Args:
        params (ParameterProvider):
            The parameter provider.
        weather_data (NASAPowerWeatherDataProvider):
            The weather data.
        agro (dict):
            The agronomy configuration.

    Returns:
        dict:
            The simulation outputs.
    """
    wofost = Wofost72_PP(params, weather_data, agro)
    wofost.run_till_terminate()
    sim_results = wofost.get_summary_output()[0]
    return sim_results
