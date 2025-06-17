from abc import ABC, abstractmethod

class BasePrior(ABC):
    """
    Abstract base class for a prior.
    Users should implement `sample` and `log_prob`.
    """
    @abstractmethod
    def sample(self, n_samples=1):
        """
        Return samples from the prior.
        """
        pass

    @abstractmethod
    def log_prob(self, params):
        """
        Return log probability of the given parameters under the prior.
        """
        pass


class BaseLikelihood(ABC):
    """
    Abstract base class for a likelihood.
    Users should implement `log_likelihood`.
    """
    @abstractmethod
    def log_likelihood(self, params):
        """
        Compute log-likelihood of the parameters given the data.
        """
        pass
