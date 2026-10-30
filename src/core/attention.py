"""
**What is Attention?**:
Attention is the mechanism that allows models to focus on relevant parts of the input
when processing sequences. Think of it as a search engine inside your neural network - given a
query, attention finds the most relevant keys and retrieves their associated values.

**The Attention Intuition**:
When you read "The cat sat on the ___", your brain automatically focuses on "cat" and "sat" to predict
"mat". This selective focus is exactly what attention mechanisms provide to neural networks.



"""

import numpy as np
rng = np.random.default_rng(7)
import math
import time
from typing import Optional, Tuple, List

from core.tensor import Tensor
from core.layers import Linear
from core.activations import Softmax

MASK_VALUE = -1e9  # Large negative value used for attention masking (becomes ~0 after softmax)


def _compute_attention_scores(Q: Tensor, K: Tensor) -> Tensor:
    """
    Compute raw attention scores via Q @ K^T

    EXAMPLE:
    >>> Q = Tensor(rng.standard_normal((1, 3, 4)))  # 3 tokens, dim=4
    >>> K = Tensor(rng.standard_normal((1, 3, 4)))
    >>> scores = _compute_attention_scores(Q, K)
    >>> print(scores.shape)  # (1, 3, 3) -- every token scored against every other
    """

    K_t = K.transpose(-2, -1)
    return Q.matmul(K_t)


def _scale_scores(scores: Tensor, d_model: int) -> Tensor:
    """
    Scale attention scores by 1/sqrt(d_model).

    EXAMPLE:
    >>> scores = Tensor(np.array([[[4.0, 8.0]]]))
    >>> scaled = _scale_scores(scores, d_model=4)
    >>> print(scaled.data)  # [[[ 2.0, 4.0]]] -- divided by sqrt(4)=2
    """

    scale_factor = 1.0 / math.sqrt(d_model)
    return scores * scale_factor


def _apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """
    Apply causal mask by setting masked positions to -infinity.

    EXAMPLE:
    >>> scores = Tensor(np.ones((1, 3, 3)))
    >>> mask = Tensor(np.tril(np.ones((1, 3, 3))))  # lower triangle
    >>> masked = _apply_mask(scores, mask)
    >>> print(masked.data[0, 0, 1])  # -1e9 (future position masked)
    """

    adder = (Tensor(np.ones_like(mask.data)) - mask) * MASK_VALUE
    return scores + adder


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Complete scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch_size, seq_len, d_model)
        K: Key tensor of shape (batch_size, seq_len, d_model)
        V: Value tensor of shape (batch_size, seq_len, d_model)
        mask: Optional causal mask, 1=allow, 0=mask (batch_size, seq_len, seq_len)

    Returns:
        output: Attended values (batch_size, seq_len, d_model)
        attention_weights: Attention matrix (batch_size, seq_len, seq_len)

    EXAMPLE:
    >>> Q = Tensor(rng.standard_normal((2, 4, 64)))
    >>> K = Tensor(rng.standard_normal((2, 4, 64)))
    >>> V = Tensor(rng.standard_normal((2, 4, 64)))
    >>> output, weights = scaled_dot_product_attention(Q, K, V)
    >>> print(output.shape)   # (2, 4, 64)
    >>> print(weights.shape)  # (2, 4, 4)
    """

    scores = _compute_attention_scores(Q, K)
    scores = _scale_scores(scores, Q.shape[-1])
    if mask is not None:
        scores = _apply_mask(scores, mask)
    softmax = Softmax()
    attention_weights = softmax(scores, dim=-1)
    output = attention_weights.matmul(V)
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel, each learning different relationships.
    This is the core component of transformer architecture.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension (d_model)
            num_heads: Number of parallel attention heads

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=512, num_heads=8)
        >>> mha.head_dim  # 64 (512 / 8)
        >>> len(mha.parameters())  # 4 linear layers * 2 params each = 8 tensors
        """

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Multi-head attention dimension mismatch\n"
                f"  embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        # Output projection to mix information across heads
        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Reshape to separate attention heads for parallel processing.


        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(rng.standard_normal((2, 10, 64)))  # batch=2, seq=10
        >>> split = mha._split_heads(x, 2, 10)
        >>> print(split.shape)  # (2, 8, 10, 8) -- 8 heads of dim 8
        """
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor, batch_size: int, seq_len) -> Tensor:
        """
        Merge attention heads back into single embedding dimension.

        EXAMPLE:
        >>> # After attention with 8 heads of dim 8:
        >>> attended = Tensor(rng.standard_normal((2, 8, 10, 8)))
        >>> merged = mha._merge_heads(attended, 2, 10)
        >>> print(merged.shape)  # (2, 10, 64) -- back to embed_dim
        """

        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (batch_size, seq_len, seq_len)

        Returns:
            output: Attended representation (batch_size, seq_len, embed_dim)

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(rng.standard_normal((2, 10, 64)))  # batch=2, seq=10, dim=64
        >>> output = mha.forward(x)
        >>> print(output.shape)  # (2, 10, 64) - same as input
        """

        # STEP 1: Extract dimensions and validate
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention input dimension mismatch\n"
                f"  Expected embed_dim={self.embed_dim}, got {embed_dim} from input shape {x.shape}\n"
            )

        #

        # STEP 2: Project to Q, K, V
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # STEP 3: Split into heads
        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)

        # STEP 4: Apply attention. Reshape mask (batch, seq_q, seq_k) to
        # (batch, 1, seq_q, seq_k) for broadcasting across heads
        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
            batch_size_mask, seq_q_mask, seq_k_mask = mask.shape
            mask_reshaped = mask.reshape(batch_size_mask, 1, seq_q_mask, seq_k_mask)

        attended, _ = scaled_dot_product_attention(Q, K, V, mask=mask_reshaped)

        # STEP 5: Merge heads back together
        concat_output = self._merge_heads(attended, batch_size, seq_len)

        # STEP 6: Apply output projection
        output = self.out_proj.forward(concat_output)

        return output

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Make MultiHeadAttention callable like attention(x)."""
        return self.forward(x, mask)

    def parameters(self) -> List[Tensor]:
        """
        Return all trainable parameters.

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> params = mha.parameters()
        >>> print(len(params))  # 8 (4 layers × 2 params each: weight + bias)
        >>> print(params[0].shape)  # (64, 64) - q_proj weight
        >>> print(params[1].shape)  # (64,) - q_proj bias
        """

        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
