import numpy as np
import torch
from tqdm import tqdm

class HierarchicalFlowTrainer:
    def __init__(self, flow_model, prior, likelihoods, device="cpu"):
        """
        Parameters:
        - flow_model: an unconditional normalizing flow (e.g., glasflow or nflows)
        - prior: instance of BasePrior
        - likelihoods: list of BaseLikelihood objects, one per event
        - device: 'cpu' or 'cuda'
        """
        self.flow = flow_model.to(device)
        self.prior = prior
        self.likelihoods = likelihoods
        self.device = device
        self.param_names = None

    def compute_weights(self, theta_samples):
        """
        Computes normalized hierarchical weights from multiple likelihoods.

        Parameters:
        - theta_samples: numpy array of shape (N, D)

        Returns:
        - normalized weights (numpy array of shape (N,))
        """
        log_probs = np.zeros(theta_samples.shape[0])

        for lh in self.likelihoods:
            event_log_probs = np.array([
                lh.log_likelihood(dict(zip(self.param_names, theta)))
                for theta in theta_samples
            ])
            log_probs += event_log_probs - np.max(event_log_probs)  # log-stability per event

        weights = np.exp(log_probs)
        return weights / np.sum(weights)

    def train(self, n_epochs, n_samples_per_epoch, optimizer, scheduler=None):
        """
        Trains the flow using importance-weighted likelihood.

        Parameters:
        - n_epochs: number of training epochs
        - n_samples_per_epoch: number of prior samples per epoch
        - optimizer: torch optimizer
        - scheduler: optional LR scheduler
        """
        self.flow.train()
        losses = []

        for epoch in range(n_epochs):
            # Sample from prior
            theta_samples_dict = self.prior.sample(n_samples_per_epoch)
            self.param_names = list(theta_samples_dict.keys())
            theta_samples_np = np.column_stack([theta_samples_dict[k] for k in self.param_names])

            # Compute weights
            weights = self.compute_weights(theta_samples_np)

            # Convert to torch tensors
            x = torch.tensor(theta_samples_np, dtype=torch.float32).to(self.device)
            w = torch.tensor(weights, dtype=torch.float32).to(self.device)

            # Compute loss
            logp = self.flow.log_prob(x).squeeze()
            loss = -(w * logp).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            losses.append(loss.item())
            print(f"[Epoch {epoch+1}/{n_epochs}] Loss: {loss.item():.6f}")

        return losses

    def sample_posterior(self, n_samples=1000, resample_with_weights=False):
        """
        Samples from the trained flow (approximate posterior).

        Parameters:
        - n_samples: number of samples to draw
        - resample_with_weights: if True, perform importance resampling using original hierarchical weights

        Returns:
        - numpy array of posterior samples, shape (n_samples, D)
        """
        self.flow.eval()
        with torch.no_grad():
            samples = self.flow.sample(n_samples).cpu().numpy()

        if resample_with_weights:
            print("Resampling flow samples using hierarchical weights...")
            weights = self.compute_weights(samples)
            resample_idxs = np.random.choice(np.arange(len(samples)), size=n_samples, p=weights)
            samples = samples[resample_idxs]

        return samples

    def sample_posterior_dict(self, n_samples=1000):
        """
        Returns posterior samples as a dictionary keyed by parameter name.
        """
        samples = self.sample_posterior(n_samples)
        return {key: samples[:, i] for i, key in enumerate(self.param_names)}
