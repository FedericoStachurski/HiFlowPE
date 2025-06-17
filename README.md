# HiFlowPE: Hierarchical Inference with Normalizing Flows

**HiFlowPE** is a Python package for hierarchical Bayesian inference using unconditional normalizing flows trained via importance weighting. It is designed for use cases such as cosmological or astrophysical population parameter estimation from multiple observed events (e.g., gravitational waves).

---

## Overview

HiFlowPE enables:

* Defining flexible **priors** and **likelihoods** using a modular object-oriented interface
* Combining data from **multiple events** into a joint likelihood using importance sampling
* Training a **normalizing flow** model on weighted prior samples to approximate the posterior
* Sampling from the trained flow to obtain posterior distributions

---

## Features

* Modular abstract base classes for `Prior` and `Likelihood`
* Built-in components: `UniformPrior`, `GaussianLikelihood`, and `NFLikelihood`
* A central `HierarchicalFlowTrainer` class for importance-weighted training
* Posterior sampling with optional resampling
* Compatible with [GlasFlow](https://github.com/glasgow-astro/glasflow) and `nflows`
* Full GPU support via PyTorch

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/HiFlowPE.git
cd HiFlowPE
conda env create -f environment.yml
conda activate hiflowpe_env
```

---

## Example Usage

```python
from HiFlowPE import HierarchicalFlowTrainer, UniformPrior, GaussianLikelihood
from glasflow.flows import RealNVP
import torch

# Define prior and per-event likelihoods
prior = UniformPrior({"H0": (50, 100), "Om": (0.2, 0.4)})
likelihoods = [GaussianLikelihood(data=d, sigma=s) for d, s in zip(data_list, sigma_list)]

# Initialize flow and optimizer
flow = RealNVP(n_inputs=2, n_transforms=5, n_neurons=64).to("cuda")
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

# Train flow on hierarchical posterior
trainer = HierarchicalFlowTrainer(flow, prior, likelihoods, device="cuda")
trainer.train(n_epochs=100, n_samples_per_epoch=50000, optimizer=optimizer)

# Sample from posterior
posterior_samples = trainer.sample_posterior(n_samples=10000, resample_with_weights=True)
```

---

## File Structure

```
HiFlowPE/
├── __init__.py
├── initial_classes.py           # BasePrior and BaseLikelihood definitions
├── model_components.py          # Example prior/likelihood implementations
├── hierarchical_flow_trainer.py # Flow training and posterior sampling logic
├── environment.yml              # Conda environment specification
```

---

## License

MIT License

---

## Author

Federico Stachurski
Postdoctoral Researcher, University of Glasgow
[Email](mailto:your-email@example.com)

