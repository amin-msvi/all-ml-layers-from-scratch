"""
STEP 2: Sampling - One Logits Row Becomes One Token

The model hands over V scores. It does not choose. Turning the scores into one
token id is the first decision the harness makes, and it is a setting, not a 
weight: the same model writes differently under a different policy.

```
logits (V,) --> / temperature --> softmax --> top-k cut --> top-p cut --> draw one id
```

**What this module covers**:
    - Greedy: take the argmax (temperature=0). Deterministic, repeats itself
    - Temperature: divide the logits before softmax. Below 1 sharpens, above 1 flattens
    - Top-k: keep the k most likely tokens and renormalise
    - Top-p (nucleus): keep the smallest set whose probability mass reaches p
    - Entropy: how much choice is left in a distribution

The random generator is always passed in. A sampler that owns a hidden rng cannot
be replayed. One that receives a seeded rng gives the same token every run.
"""

from typing import Optional

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to probabilities along the last axis.

    EXAMPLE:
    >>> softmax(np.array([1.0, 2.0, 3.0]))
    array([0.09003057, 0.24472847, 0.66524096])
    """
    # Subtract the max for numerical stability (same trick as core.activations.Softmax)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """
    Keep the k largest probabilities, zero the rest, renormalize.

    EXAMPLE:
    >>> top_k_filter(np.array([0.5, 0.3, 0.2]), k=2)
    array([0.625, 0.375, 0.   ])
    """
    keep = np.argsort(probs)[-k:]  # indices of the k largest

    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return filtered / filtered.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """
    Keep the smallest set of tokens whose total probability reaches p.

    Unlike top-k the size of the kept set adapts to the row: a flat row
    needs many tokens to reach p, a sharp row reaches it with one.

    EXAMPLE:
    >>> top_p_filter(np.array([0.5, 0.3, 0.2]), p=0.7)
    array([0.625, 0.375, 0.   ])
    """
    order = np.argsort(probs)[::-1]  # most likely first
    cumulative = np.cumsum(probs[order])

    # Shortest prefix with cumulative mass >= p
    num_kept = int(np.searchsorted(cumulative, p)) + 1
    keep = order[:num_kept]

    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return filtered / filtered.sum()


def next_token_probs(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> np.ndarray:
    """
    The distribution the next token is drawn from, after the decoding policy.

    Args:
        logits: Scores of shape (V,) from the model
        temperature: 0 = greedy, below 1 = sharper, above 1 = flatter
        top_k: Optional number of most likely tokens to keep
        top_p: Optional probability mass to keep

    Returns:
        probabilities of shape (V,) that sum to 1

    EXAMPLE:
    >>> next_token_probs(np.array([1.0, 2.0, 3.0]), temperature=0)
    array([0., 0., 1.])
    """
    logits = np.asarray(logits, dtype=np.float64)

    # Greedy is the limit temperature -> 0: all mass on the argmax
    if temperature == 0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs

    probs = softmax(logits / temperature)
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    return probs


def sample(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """
    Choose one token id from a logits row.

    EXAMPLE:
    >>> rng = np.random.default_rng(1)
    >>> token = sample(np.array([1.0, 2.0, 3.0]), rng, temperature=0.8, top_k=2)
    >>> token in (1, 2)
    True
    """
    probs = next_token_probs(logits, temperature, top_k, top_p)
    return int(rng.choice(len(probs), p=probs))


def entropy(probs: np.ndarray) -> float:
    """
    Entropy in nats: 0 when one token has all the mass, ln(V) when uniform.

    EXAMPLE:
    >>> entropy(np.array([0.5, 0.5]))  # ln(2)
    0.6931471805599453
    """
    kept = probs[probs > 0]
    return float(np.sum(kept * np.log(1.0 / kept)))
