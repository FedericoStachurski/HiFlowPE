# model_components.py

from .initial_classes import BasePrior, BaseLikelihood
import numpy as np
import torch
from scipy.stats import uniform, norm


##################################################
################## Priors ########################
##################################################

class UniformPrior(BasePrior):
    def __init__(self, param_ranges):
        self.ranges = param_ranges

    def sample(self, n_samples=1):
        return {
            key: np.random.uniform(low, high, size=n_samples)
            for key, (low, high) in self.ranges.items()
        }

    def log_prob(self, params):
        logp = 0
        for key, val in params.items():
            low, high = self.ranges[key]
            if not (low <= val <= high):
                return -np.inf
            logp += uniform(loc=low, scale=high - low).logpdf(val)
        return logp


##################################################
################## Likelihoods ################### 
##################################################

class GaussianLikelihood(BaseLikelihood):
    def __init__(self, data, sigma):
        self.data = data
        self.sigma = sigma

    def log_likelihood(self, params):
        logL = 0.0
        for key in self.data:
            mu = params[key]
            obs = self.data[key]
            sig = self.sigma[key]
            logL += norm.logpdf(obs, loc=mu, scale=sig)
        return logL



class NFLikelihood(BaseLikelihood):
    """
    Wraps a trained normalizing flow as a likelihood model.
    Assumes that the NF models p(x | θ) or p(data | params).
    """

    def __init__(self, nf_model, condition_keys=None, device="cpu"):
        """
        nf_model: a trained flow model (e.g., glasflow FlowWrapper)
        condition_keys: optional list of keys to extract conditional inputs
        """
        self.flow = nf_model.to(device)
        self.device = device
        self.condition_keys = condition_keys

    def log_likelihood(self, params):
        """
        Compute log-likelihood using the flow model.
        If the model is conditional, `params` must contain both data and context.
        """
        # Convert params dict to tensors
        x = torch.tensor([params[k] for k in self.flow.data_keys], dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.condition_keys:
            context = torch.tensor([params[k] for k in self.condition_keys], dtype=torch.float32).unsqueeze(0).to(self.device)
            log_prob = self.flow.log_prob(x, context)
        else:
            log_prob = self.flow.log_prob(x)

        return log_prob.item()
