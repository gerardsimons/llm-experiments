import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy


def log_margin(logprobs: dict) -> float:
    """
    The difference in logspace between the top choice and the second choice.
    """
    vals = list(logprobs.values())
    vals.sort(reverse=True)
    return vals[0] - vals[1]



def predictive_entropy(logprobs: dict) -> float:
    """
    Compute predictive entropy from log-probabilities.
    """
    lp = np.array(list(logprobs.values()))

    # Top tokens may have been truncated or have other numerical issues so they no longer sum to 1
    # Normalize in log space (stable)
    log_p = lp - logsumexp(lp)

    # Then convert to probabilities
    p = np.exp(log_p)

    # Entropy is best computed in probability space
    return entropy(p)


