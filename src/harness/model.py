"""
STEP 1: Model - The Stateless Core of the Harness

A Harness is everything around a model, so the first step is to pin down what sits
is the middle. GPT is a pure function from token ids to next-token logits

```
IDs (B, T)  -->  GPT.forward  -->  logits (B, T, V)
                 weights only, nothing is kept between two calls
```

The model has:
    - NO memory of a conversation
    - NO stop rule
    - NO way to act.
In Harness, every layer on *around* model adds one of those around this function,
and non of them changes it.

---

**What this module covers**:
    - V1: The tiny GPT every demo runs on:
        - 94-character corpus
        - CharTokenizer (V = 18)
        - GPT(V=18, E=16, L=2, H=2, T=16): 7,424 parameters
        - Training it once (Adam, 401 steps) and keeping the weights on disk
        - next_token_logits(): the single seam between harness and model

**CONTRACT: What the harness needs from a model**:
    - model.forward(Tensor (B, T)) -> Tensor (B, T, V)
    - model.max_seq_len: the window size T
Any object with these two can replace the tiny GPT without touching the harness.
"""

import os
from typing import List, Tuple

import numpy as np

from core.autograd import no_grad
from core.losses import CrossEntropyLoss
from core.optimizers import Adam
from core.tensor import Tensor
from core.tokenization import CharTokenizer, Tokenizer
from core.transformers import GPT

# The whole training text: four sentences, 94 characters, 17 distinct ones
CORPUS = (
    "the cat sat on the mat . the dog sat on the log . "
    "the cat ate the rat . the dog ate the bun . "
)

# Trained weights are written here once: delete the file to train again
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_gpt.npz")

def build_tokenizer() -> CharTokenizer:
    """
    Build the character tokenizer of the corpus.

    EXAMPLE:
    >>> tokenizer = build_tokenizer()
    >>> tokenizer.vocab_size
    18
    >>> tokenizer.encode("the ")
    [16, 9, 7, 1]
    """
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([CORPUS])
    return tokenizer


def make_windows(ids: List[int], T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut a token sequence into training windows: inputs and next-token targets.

    Args:
        ids: The whole corpus as token ids
        T: Window length

    Returns:
        X: Inputs of shape (N, T)
        Y: Targets of shape (N, T), X shifted left by one position

    EXAMPLE:
    >>> X, Y = make_windows(tokenizer.encode(CORPUS), T=16)
    >>> X.shape  # 94 - 16 = 78 windows
    (78, 16)
    """
    num_windows = len(ids) - T
    X = np.array(ids[i:i + T] for i in range(num_windows))
    Y = np.array(ids[i+1:i+T + 1] for i in range(num_windows))
    return X, Y


def train(
    model: GPT, 
    X: np.ndarray,
    Y: np.ndarray,
    steps: int = 401,
    lr: float = 0.01,
) -> List[float]:
    """
    Train the model on all windows as one batch and return the loss per step.

    EXAMPLE:
    >>> losses = train(model, X, Y)
    >>> print(f"{losses[0]:.3f} -> {losses[-1]:.3f}")  # 3.473 -> 0.131
    """
    params = model.parameters()

    # Layers create their weights without requires_grad, the same reason
    # Trainer.__init__ sets it explicitely
    for param in params:
        param.requires_grad = True

    optimizer = Adam(params, lr=lr)
    loss_fn = CrossEntropyLoss()
    inputs = Tensor(X)
    targets = Tensor(Y.reshape(-1))  # (N * T,)

    losses = []
    for _ in range(steps):
        logits = model.forward(inputs)  # (N, T, V)
        loss = loss_fn.forward(logits.reshape(-1, model.vocab_size), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.data))

    return losses


def save_weights(model: GPT, path: str) -> None:
    """Write every parameter array to one .npz file, in parameters() order."""
    np.savez(path, *[param.data for param in model.parameters()])


def load_weights(model: GPT, path: str) -> None:
    """Read the parameter arrays back into a model of the same architecture."""
    with np.load(path) as arrays:
        for i, param in enumerate(model.parameters()):
            param.data = arrays[f"arr_{i}"]


def tiny_gpt() -> Tuple[GPT, CharTokenizer]:
    """
    Return the trained tiny GPT and its tokenizer.

    Loads the weights from WEIGHT_PATH when the file exists, otherwise
    trains for about 10 seconds and writes the file.

    EXAMPLE:
    >>> model, tokenizer = tiny_gpt()
    >>> logits = next_token_logits(model, tokenizer.encode(" cat sat on the "))
    >>> tokenizer.decode([int(np.argmax(logits))])
    """
    tokenizer = build_tokenizer()
    model = GPT(
        tokenizer.vocab_size,
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )

    if os.path.exists(WEIGHTS_PATH):
        load_weights(model, WEIGHTS_PATH)
    else:
        X, Y = make_windows(tokenizer.encode(CORPUS), model.max_seq_len)
        train(model, X, Y)
        save_weights(model, WEIGHTS_PATH)

    return model, tokenizer


def next_token_logits(model, ids: List[int]) -> np.ndarray:
    """
    Run the model on one window and return the logits row for the next token.

    This is the only place the harness calls the model. The caller makes sure
    that len(ids) <= model.max_seq_len.

    Args:
        model: Any object with forward(Tensor (B, T)) -> Tensor (B, T, V)
        ids: Token ids of the window, oldest first

    Returns:
        Logits of shape (V,): one score per vocabulary entry

    EXAMPLE:
    >>> logits = next_token_logits(model, tokenizer.encode(" the "))
    >>> logits.shape
    (18,)
    """

    # Inference only: no graph is needed
    with no_grad():
        logits = model.forward(Tensor(np.array(ids)))  # (1, T, V)

    # Row T - 1 scores the token that comes after the window
    return logits.data[0, -1, :]

