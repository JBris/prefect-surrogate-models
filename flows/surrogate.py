#!/usr/bin/env python

######################################
# Imports
######################################

import mlflow
from prefect import task
import torch
import gpytorch

######################################
# Class
######################################

class MultitaskVariationalGPModel(gpytorch.models.ApproximateGP):
    """
    A multitask variational Gaussian process model.

    Args:
        gpytorch (gpytorch.models.ApproximateGP):
            An approximate, variational Gaussian process.
    """

    def __init__(self, n_col: int, num_latents: int, num_tasks: int):
        """
        Constructor.

        Args:
            n_col (int):
                The number of columns.
            num_latents (int):
                The number of latent variables.
            num_tasks (int):
                The number of tasks for multitask learning.
        """
        inducing_points = torch.rand(num_latents, n_col, n_col)

        variational_distribution = (
            gpytorch.variational.MeanFieldVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([num_latents])
        )
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=2.5, batch_shape=torch.Size([num_latents]), ard_num_dims=n_col
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        The forward pass.

        Args:
            x (torch.Tensor):
                The input data.

        Returns:
             gpytorch.distributions.MultivariateNormal:
                A multivariate normal random variable.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
@task
def load_surrogate(
    name: str
) -> tuple[MultitaskVariationalGPModel,  gpytorch.likelihoods.MultitaskGaussianLikelihood]:
    """
    Load the surrogate model.

    Args:
        name (str): 
            The surrogate model name.

    Returns:
        tuple[MultitaskVariationalGPModel,  
        gpytorch.likelihoods.MultitaskGaussianLikelihood]: The surrogate and likelihood.
    """
    model_uri_prefix = "models:/"
    model_uri_suffix = "/latest"

    model = mlflow.pytorch.load_model(
        model_uri = f"{model_uri_prefix}{name}{model_uri_suffix}"
    )

    likelihood = mlflow.pytorch.load_model(
        model_uri = f"{model_uri_prefix}{name}_likelihood{model_uri_suffix}"
    )

    return model, likelihood