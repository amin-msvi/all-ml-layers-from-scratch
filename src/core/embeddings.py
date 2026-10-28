"""
Complete Embedding System Architecture

The production embedding layer that powers modern transformers combines multiple components into an efficient, flexible pipeline.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMPLETE EMBEDDING SYSTEM: Token + Position → Position-Aware Representations │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ INPUT: Token IDs [1, 42, 7, 99]                                              │
│         │                                                                    │
│         ├─ STEP 1: TOKEN EMBEDDING LOOKUP                                    │
│         │  ┌─────────────────────────────────────────────────────────┐       │
│         │  │   Token Embedding Table (vocab_size × embed_dim)        │       │
│         │  │                                                         │       │
│         │  │   ID 1  → [0.1,  0.4, -0.2, ...]  (semantic features)   │       │
│         │  │   ID 42 → [0.7, -0.2,  0.1, ...]  (learned meaning)     │       │
│         │  │   ID 7  → [-0.3, 0.1,  0.5, ...]  (dense vector)        │       │
│         │  │   ID 99 → [0.9, -0.1,  0.3, ...]  (context-free)        │       │
│         │  └─────────────────────────────────────────────────────────┘       │
│         │                                                                    │
│         ├─ STEP 2: POSITIONAL ENCODING (Choose Strategy)                     │
│         │  ┌─────────────────────────────────────────────────────────┐       │
│         │  │ Strategy A: Learned PE                                  │       │
│         │  │   pos 0 → [trainable vector] (learns patterns)          │       │
│         │  │   pos 1 → [trainable vector] (task-specific)            │       │
│         │  │   pos 2 → [trainable vector] (fixed max length)         │       │
│         │  │                                                         │       │
│         │  │ Strategy B: Sinusoidal PE                               │       │
│         │  │   pos 0 → [sin/cos pattern] (mathematical)              │       │
│         │  │   pos 1 → [sin/cos pattern] (no parameters)             │       │
│         │  │   pos 2 → [sin/cos pattern] (infinite length)           │       │
│         │  │                                                         │       │
│         │  │ Strategy C: No PE                                       │       │
│         │  │   positions ignored (order-agnostic)                    │       │
│         │  └─────────────────────────────────────────────────────────┘       │
│         │                                                                    │
│         ├─ STEP 3: ELEMENT-WISE ADDITION                                     │
│         │  ┌─────────────────────────────────────────────────────────┐       │
│         │  │ Token + Position = Position-Aware Representation        │       │
│         │  │                                                         │       │
│         │  │ [0.1, 0.4, -0.2] + [pos0] = [0.1+p0, 0.4+p0, ...]       │       │
│         │  │ [0.7, -0.2, 0.1] + [pos1] = [0.7+p1, -0.2+p1, ...]      │       │
│         │  │ [-0.3, 0.1, 0.5] + [pos2] = [-0.3+p2, 0.1+p2, ...]      │       │
│         │  │ [0.9, -0.1, 0.3] + [pos3] = [0.9+p3, -0.1+p3, ...]      │       │
│         │  └─────────────────────────────────────────────────────────┘       │
│         │                                                                    │
│         ├─ STEP 4: OPTIONAL SCALING (Transformer Convention)                 │
│         │  ┌─────────────────────────────────────────────────────────┐       │
│         │  │ Scale by √embed_dim for gradient stability              │       │
│         │  │ Helps balance token and position magnitudes             │       │
│         │  └─────────────────────────────────────────────────────────┘       │
│         │                                                                    │
│         └─ OUTPUT: Position-Aware Dense Vectors                              │
│            Ready for attention mechanisms and transformers!                  │
│                                                                              │
│ INTEGRATION FEATURES:                                                        │
│ • Flexible position encoding (learned/sinusoidal/none)                       │
│ • Efficient batch processing with variable sequence lengths                  │
│ • Memory optimization (shared position encodings)                            │
│ • Production patterns (matches PyTorch/HuggingFace)                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

**Why this architecture works**: By separating token semantics from positional information, 
the model can learn meaning and order independently, then combine them optimally for the specific task.
```
"""

import numpy as np
rng = np.random.default_rng(7)
import math
from typing import List, Optional, Tuple

from core.tensor import Tensor

# Enable autograd for gradient tracking (required for learnable embeddings)
from core.autograd import Function, enable_autograd
enable_autograd()

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


class EmbeddingBackward(Function):
    """
    Gradient computation for embedding lookup operation.

    **Mathematical Rule:** 
        If Y = Embedding[indices], then:
        ∂Loss/∂Embedding[i] = sum of all gradients where index==i

    When the same token appears multiple times in a sequence (like word "the"),
    its embedding row accumulates gradients from every position.

    Embedding lookup is a 'gather' operation. The backward
    is a 'scatter' operation that accumulates gradients to the embedding weights.
    """
    def __init__(self, weight, indices):
        super().__init__(weight)
        self.indices = indices

    def apply(self, grad_output):
        """
        Compute gradient for embedding lookup.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for weight tensor

        **Mathematical Foundation:**
        - ∂(Embedding[indices])/∂Embedding = scatter gradients to selected rows
        - Multiple indices can point to same embedding -> gradients accumulate 
        """

        weight, = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:
            # Initialize gradient with zeros
            grad_weight = np.zeros_like(weight.data)

            # Scatter gradients back to embedding weights
            # np.add.at accumulates gradients for repeated indices
            indices_flat = self.indices.dota.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight, )


class Embedding:
    """
    Learnable embedding layer that maps token indices to dense vectors.

    This is the fundamental building block for converting discrete tokens
    into continuous representations that neural network can process.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initalize embedding layer with Xavier-uniform weights.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            embed_dim: Dimension of embedding vectors
        """

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier initialization for better gradient flow
        limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Tensor(
            rng.uniform(-limit, limit, (vocab_size, embed_dim))
        )

    def forward(self, indices: Tensor) -> Tensor:
        """
        Forward pass: lookup embeddings for given indices.

        Args:
            indices: Token indices of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Embedded vectors of shape (*indices.shape, embed_dim)
        """

        if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
            min_idx = int(np.min(indices.data))
            max_idx = int(np.max(indices.data))
            raise ValueError(
                "Embedding index out of range for vocabulary size {self.vocab_size}\n"
                f"Found indices: min={min_idx}, max={max_idx} (valid range: 0 to {self.vocab_size - 1})"
            )

        # Perform embedding lookup using advanced indexing
        # This is equivalent to one-hot multiplication but much more efficient
        embedded = self.weight.data[indices.data.astype(int)]

        result = Tensor(embedded)

        # Attach gradient function for backpropagation
        # EmbeddingBackward handles spare gradient accumulation
        if self.weight.requires_grad:
            result.requires_grad = True
            result._grad_fn = EmbeddingBackward(self.weight, indices)

        return result

    def __call__(self, indices: Tensor) -> Tensor:
        """Allows the embedding to be called like a function."""
        return self.forward(indices)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.weight]

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


class PositionalEncoding:
    """
    Learnable positional encoding layer.

    Adds trainable position-specific vectors to token embeddings,
    allowing the model to learn positional patterns specific to the task.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Initialize learnable positional encoding.

        Args:
            max_seq_len: Maximum sequence length to support
            embed_dim: Embedding dimension (must match token embedding)
        """
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Initiate position embedding matrix
        # Smaller initialization than token embedding since there are additive
        limit = math.sqrt(2.0 / embed_dim)
        self.position_embeddings = Tensor(
            rng.uniform(-limit, limit, (max_seq_len, embed_dim))
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, embed_dim)

        Returns:
            Position-encoded embeddings of same shape
        """

        if len(x.shape) == 2:
            raise ValueError(
                "Expected 3D input (batch, seq, embed), got 2D: {x.shape}\n"
            )
        elif len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got {len(x.shape)}D: {x.shape}\n"
            )

        batch_size, seq_len, embed_dim = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeds maximum: {seq_len} > {self.max_seq_len}\n"
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f"PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n"
            )

        # Slice position embeddings for this sequence length using Tensor slicing
        pos_embeddings = self.position_embeddings[:seq_len]  # (seq_len, embed_dim)

        # Reshape to add batch dimension: (1, seq_len, embed_dim)
        pos_data = pos_embeddings.data[np.newaxis, :, :]
        pos_embeddings_batched = Tensor(pos_data)

        # Add positional information
        result = x + pos_embeddings_batched

        return result

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the positional encoding to be called like a function."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.position_embeddings]

    def __repr__(self):
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"


def _compute_sinusoidal_table(max_len: int, embed_dim: int) -> np.ndarray:
    """
    Compute the raw sinusoidal positional encoding table as a numpy array.

    This helper builds the (max_len, embed_dim) table of sin/cos values.
    using the formula from "Attention Is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))

    EXAMPLE:
    >>> table = _compute_sinusoidal_table(4, 8)
    >>> table.shape
    (4, 8)
    >>> table[0, 0]  # sin(0) = 0.0
    0.0
    >>> table[0, 1]  # cos(0) = 1.0
    1.0
    """

    # Create position indices [0, 1, 2, ..., max_len-1]
    position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]  # (max_len, 1)

    # Create dimension indices for calculating frequencies
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) *
        -(math.log(10000.0) / embed_dim)
    ) # (embed_dim // 2,)

    # Initialize the positional encoding matrix
    pe = np.zeros((max_len, embed_dim), dtype=np.float32)

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    if embed_dim % 2 == 1:
        # Handle odd embed_dim by only filling available positions
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)

    return pe


def create_sinusoidal_embeddings(max_seq_len: int, embed_dim: int) -> Tensor:
    """
    Create sinusoidal positional encodings as used in "Attention Is All You Need".

    These fixed encodings use sine and cosine functions to create unique
    positional patterns that don't require training and can extrapolate
    to longer sequences than seen during training.

    EXAMPLE:
    >>> pe = create_sinusoidal_embeddings(512, 64)
    >>> print(pe.shape)
    (512, 64)
    >>> # Position 0: [0, 1, 0, 1, 0, 1, ...] (sin(0)=0, cos(0)=1)
    >>> # Each position gets unique trigonometric signature
    """

    ### BEGIN SOLUTION
    pe = _compute_sinusoidal_table(max_seq_len, embed_dim)
    return Tensor(pe)


class EmbeddingLayer:
    """
    Complete embedding system combining token and positional embeddings.

    This is the production-ready component that handles the full embedding
    pipeline used in transformers and other sequence models.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        pos_encoding: str = 'learned',
        scale_embeddings: bool = False,
    ):
        """
        Initialize complete embedding system.

        EXAMPLE:
        >>> layer = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding='learned')
        >>> layer.token_embedding  # Embedding(vocab_size=100, embed_dim=64)
        >>> layer.pos_encoding     # PositionalEncoding(max_seq_len=512, embed_dim=64)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding
        self.scale_embeddings = scale_embeddings

        # Token embedding layer
        self.token_embedding = Embedding(vocab_size, embed_dim)

        # Positional encoding
        if pos_encoding == 'learned':
            self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        elif pos_encoding == 'sinusoidal':
            # Create fixed sinusoidal encoding (no parameters)
            self.pos_encoding = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        elif pos_encoding is None:
            self.pos_encoding = None
        else:
            raise ValueError(
                f"Unknown positional encoding type: '{pos_encoding}'\n"
            )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass through complete embedding system.

        EXAMPLE:
        >>> layer = EmbeddingLayer(vocab_size=100, embed_dim=64)
        >>> tokens = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> output = layer.forward(tokens)
        >>> output.shape
        (2, 3, 64)
        """

        # Handle 1D input by adding batch dimension
        if len(tokens.shape) == 1:
            # NOTE: Tensor reshape preserves gradients
            tokens = tokens.reshape(1, -1)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Get token embeddings
        token_embeds = self.token_embedding.forward(tokens)

        # Scale embeddings if requested (transformer convension)
        if self.scale_embeddings:
            scale_factor = math.sqrt(self.embed_dim)
            token_embeds = token_embeds * scale_factor  # Use Tensor multiplication to preserve gradients

        # Add positional encoding
        if self.pos_encoding_type == 'learned':
            # Use learnable positional encoding
            output = self.pos_encoding.forward(token_embeds)
        elif self.pos_encoding_type == 'sinusoidal':
            # Use fixed sinusoidal encoding (not learnable)
            batch_size, seq_len, embed_dim = token_embeds.shape
            pos_embeddings = self.pos_encoding[:seq_len]  # Slice using Tensor slicing

            # Reshape to add batch dimension
            pos_data = pos_embeddings.data[np.newaxis, :, :]
            pos_embeddings_batched = Tensor(pos_data)  # Sinusoidal are fixed

            output = token_embeds + pos_embeddings_batched
        else:
            # No positional encoding
            output = token_embeds

        # Remove batch dimension if it was added
        if squeeze_batch:
            # Use Tensor slicing
            output = output[0]

        return output

    def __call__(self, tokens: Tensor) -> Tensor:
        """Allows the embedding layer to be called like a function."""
        return self.forward(tokens)

    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        params = self.token_embedding.parameters()
        if self.pos_encoding_type == 'learned':
            params.extend(self.pos_encoding.parameters())
        return params

    