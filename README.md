# Prefect Surrogate Models

[![Validate Pipeline](https://github.com/JBris/prefect-surrogate-models/actions/workflows/validation.yaml/badge.svg?branch=main)](https://github.com/JBris/prefect-surrogate-models/actions/workflows/validation.yaml) [![Generate Documentation](https://github.com/JBris/prefect-surrogate-models/actions/workflows/docs.yaml/badge.svg)](https://github.com/JBris/prefect-surrogate-models/actions/workflows/docs.yaml) [![pages-build-deployment](https://github.com/JBris/prefect-surrogate-models/actions/workflows/pages/pages-build-deployment/badge.svg?branch=gh-pages)](https://github.com/JBris/prefect-surrogate-models/actions/workflows/pages/pages-build-deployment)

Website: [Prefect Surrogate Models](https://jbris.github.io/prefect-surrogate-models/)

*Demonstrating the use of Prefect to orchestrate the creation of machine learning surrogate models as applied to mechanistic crop models.*

# Table of contents

- [Prefect Surrogate Models](#prefect-surrogate-models)
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [The Prefect pipeline](#the-prefect-pipeline)
- [Python Environment](#python-environment)
  - [MLOps](#mlops)

# Introduction

The purpose of this project is to provide a simple demonstration of how to construct a Prefect pipeline, with MLOps integration, to orchestrate the creation of machine learning surrogate models as applied to mechanistic crop models. 

We use this machine learning model (a Gaussian process) to support various downstream model calibration tasks. In the example here, we perform global optimisation. [Note that the demonstrated approach is different from Bayesian optimisation.](https://botorch.org/docs/introduction)

[Our mechanistic crop model is WOFOST, as implemented in the Python Crop Simulation Environment library.](https://pcse.readthedocs.io)

[For building Gaussian processes, we use GPyTorch.](https://gpytorch.ai/)

[For performing optimisation, we use Optuna.](https://optuna.org/)

# The Prefect pipeline

[Prefect has been included to orchestrate the surrogate modelling pipeline.](https://www.prefect.io/)

The pipeline is composed of the following steps:

1. Make use of Latin Hypercube sampling to draw from the parameter space and construct a design matrix.
2. Run the WOFOST model *n* times for each sampled parameter set.
3. Train a variational Gaussian process to map the parameter sets against the WOFOST simulation outputs.
4. Perform parameter optimisation using the Tree-Structured Parzen Estimator algorithm. Rather than directly executing WOFOST during the optimisation procedure, we instead perform optimisation on the Gaussian process.

[Run prefect.sh to run the full pipeline.](scripts/prefect.sh)

[The results of the pipeline can be accessed from the output directory.](data/output)

# Python Environment

[Python dependencies are specified in this requirements.txt file.](services/python/requirements.txt). 

These dependencies are installed during the build process for the following Docker image: ghcr.io/jbris/prefect-surrogate-models:1.0.0

Execute the following command to pull the image: *docker pull ghcr.io/jbris/prefect-surrogate-models:1.0.0*

## MLOps

* [A Docker compose file has been provided to launch an MLOps stack.](docker-compose.yml)
* [See the .env file for Docker environment variables.](.env)
* [The docker_up.sh script can be executed to launch the Docker services.](scripts/docker_up.sh)
* [DVC is included for data version control.](https://dvc.org/)
* [MLFlow is available for experiment tracking.](https://mlflow.org/)
* [MinIO is available for storing experiment artifacts.](https://min.io/)
