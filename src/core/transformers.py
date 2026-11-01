import numpy as np
rng = np.random.default_rng(7)

from core.activations import GELU
from core.attention import MultiHeadAttention
from core.autograd import Function
from core.embeddings import EmbeddingLayer
from core.layers import Linear

# Import from previous modules - following proper dependency chain
from core.tensor import Tensor

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create a causal (autoregressive) attention mask.

    This mask ensures that position `i` can only attend to positions `j` where `j <= i`.
    Essential for autoregressive language models like GPT.

    Args:
        seq_len: Length of the sequence

    Returns:
        Tensor of shape (1, seq_len, seq_len) with:
        - 1.0 for positions that CAN be attended to (lower triangle)
        - 0.0 for positions that CANNOT be attended to (upper triangle)

    Example:
        For seq_len=4, creates:

         >>> mask = create_causal_mask(seq_len=4)
         >>> output = attention(x, mask=mask)
         >>> output
         [[1, 0, 0, 0],
          [1, 1, 0, 0],
          [1, 1, 1, 0],
          [1, 1, 1, 1]]
    """

    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])  # Add batch dimension



class _LayerNormBackward(Function):
    """
    Gradient computation for the full layer normalization operation.

    Computes gradients for x, gamma, and beta in one pass.
    output = gamma * ((x - mean) / std) + beta

    The gradient for x uses the standard LayerNorm formula:
        dx = (gamma/std) * (grad - mean(grad) - normalized * mean(grad * normalized))
    """

    def __init__(self, x, gamma, beta, normalized_data, std_data):
        """Initialize with forward pass values needed for gradient computation."""
        super().__init__(x, gamma, beta)
        self.normalized_data = normalized_data
        self.std_data = std_data

    def apply(self, grad_output):
        """Compute gradients for LayerNorm (x, gamma, beta)."""
        x, gamma, beta = self.saved_tensors

        grad_x = grad_gamma = grad_beta = None
        normalized = self.normalized_data
        std_data = self.std_data

        # Gradient for beta: sum over all dims except last
        if isinstance(beta, Tensor) and beta.requires_grad:
            # Sum over batch and sequence dimensions
            grad_beta = grad_output.copy()
            while grad_beta.ndim > 1:
                grad_beta = grad_beta.sum(axis=0)

        # Gradient for gamma: sum of (grad_output * normalized) over batch/seq dims
        if isinstance(gamma, Tensor) and gamma.requires_grad:
            grad_gamma = (grad_output * normalized).copy()
            while grad_gamma.ndim > 1:
                grad_gamma = grad_gamma.sum(axis=0)

        # Gradient for x: full LayerNorm backward formula
        if isinstance(x, Tensor) and x.requires_grad:
            # grad flowing through gamma: grad_output * gamma
            gamma_data = gamma.data if isinstance(gamma, Tensor) else gamma
            grad_norm = grad_output * gamma_data

            mean_grad = np.mean(grad_norm, axis=-1, keepdims=True)
            mean_grad_norm = np.mean(grad_norm * normalized, axis=-1, keepdims=True)
            grad_x = (1.0 / std_data) * (grad_norm - mean_grad - normalized * mean_grad_norm)

        return (grad_x, grad_gamma, grad_beta)


class LayerNorm:
    """
    Layer Normalization for transformer blocks.

    Normalizes across the feature dimension (last axis) for each sample independently,
    unlike batch normalization which normalizes across the batch dimension.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize LayerNorm with learnable parameters.

        EXAMPLE:
        >>> ln = LayerNorm(512)  # For 512-dimensional embeddings
        >>> x = Tensor(rng.standard_normal((2, 10, 512)))  # (batch, seq, features)
        >>> normalized = ln.forward(x)
        >>> # Each (2, 10) sample normalized independently across 512 featurs
        """

        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters: scale and shift
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)  # Scale parameter
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)  # Shift parameter

    def forward(self, x):
        """
        Apply layer normalization.

        MATHEMATICAL FORMULA:
            y = (x - μ) / σ * γ + β
        where μ = mean(x), σ = sqrt(var(x) + ε)
        """

        # Compute statistics across last dimension (features)
        mean_data = np.mean(x.data, axis=-1, keepdims=True)

        # Compute variance: E[(x - μ)²]
        diff = x.data - mean_data
        variance = np.mean(diff * diff, axis=-1, keepdims=True)

        # Normalize: (x - mean) / sqrt(variance + eps)
        std_data = np.sqrt(variance + self.eps)
        normalized_data = diff / std_data

        # Apply learnable transformation: gamma * normalized + beta
        output_data = self.gamma.data * normalized_data + self.beta.data
        output = Tensor(output_data)

        # Attach gradient function for full LayerNorm backward
        if x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad:
            output.requires_grad = True
            output._grad_fn = _LayerNormBackward(
                x, self.gamma, self.beta, normalized_data, std_data
            )

        return output

    def __call__(self, x):
        """Allows the layer norm to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return learnable parameters."""
        return [self.gamma, self.beta]


class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Network) for transformer blocks.

    Standard pattern: Linear -> GELU -> Linear with expansion ration of 4:1.
    This provides the non-linear transformation in each transformer block.
    """

    def __init__(self, embed_dim, hidden_dim=None, dropout_prob=0.1):
        """
        Initialize MLP with two linear layers.

        EXAMPLE:
        >>> mlp = MLP(512)  # Will create 512 -> 2048 -> 512 network
        >>> x = Tensor(rng.standard_normal((2, 10, 512)))
        >>> output = mlp.forard(x)
        >>> assert output.shape == (2, 10, 512)
        """

        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Standard 4x expansion

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Two-layer feed-forward network
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()
        self.linear2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass through MLP.

        COMPUTATION FLOW:
        x -> Linear -> GELU -> Linear -> output
        """
        # First linear layer with expansion
        hidden = self.linear1.forward(x)

        # GELU activation
        hidden = self.gelu.forward(hidden)

        # Second linear layer back to original size
        output = self.linear2.forward(hidden)

        return output

    def __call__(self, x):
        """Allows the MLP to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params


class TransformerBlock:
    """
    Complete Transformer Block with self-attention, MLP, and residual connetions.

    This si the core building block of GPT and other transformer models.
    Each block processes the input sequence and passes it to the next block.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, ff_dim=None, dropout_prob=0.1):
        """
        Initialize a complete transformer block.

        TRANSFORMER BLOCK ARCHITECTURE:
        x -> LayerNorm -> MultiHeadAttention -> +(residual) ->
             LayerNorm -> MLP -> +(residual) -> output

        EXAMPLE:
        >>> block = TransformerBlock(embed_dim=512, num_heads=8)
        >>> x = Tensor(rng.standard_norm((2, 10, 512)))  # (batch, seq, embed)
        >>> output = block.forward(x)
        >>> assert output.shape == (2, 10, 512)
        """

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Layer normalizations (pre-norm architecture)
        self.ln1 = LayerNorm(embed_dim)  # Before attention
        self.ln2 = LayerNorm(embed_dim)  # Before MLP

        # Feed-forward network
        # Support both mlp_ratio and explicit ff_dim for backward compatibility
        if ff_dim is not None:
            hidden_dim = ff_dim
        else:
            hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = MLP(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.

        COMPUTATION FLOW:
        x -> ln1 -> attention -> + x -> ln2 -> mlp -> + -> output

        RESIDUAL CONNECTION:
            These are crucial for training deep network. They allow
            gradients to flow directly through the network during
            backpropagation.
        """

        # First sub-layer: Multi-head self-attention with residual connection
        # Pre-norm: LayerNorm before attention
        normed1 = self.ln1.forward(x)
        # Self-attention: QUERY, KEY, VALUE are all the same (normed1)
        attention_out = self.attention.forward(normed1, mask)

        # Residual connection
        x = x + attention_out

        # Second sub-layer: MLP with residual connection
        # Pre-norm: LayerNorm before MLP
        normed2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(normed2)

        # Residual connection
        output = x + mlp_out

        return output

    def __call__(self, x, mask=None):
        """Allows the transformer block to be called like a function."""
        return self.forward(x, mask)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params



class GPT:
    """
    Complete GPT (Generative Pre-trained Transformer) model.

    This combines embeddings, positional encoding, multiple transformer blocks,
    and a language modeling head for text generation.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024):
        """
        Initialize complete GPT model.

        GPT ARCHITECTURE:
            tokens -> embedding -> + pos_embedding ->
                transformer_blocks -> layer_norm -> lm_head -> logits

        EXAMPLE:
        >>> model = GPT(vocab_size=1000, embed_dim=256, num_layers=6, num_heads=8)
        >>> tokens = Tensor(rng.integers(0, 1000, (2, 10)))  # (batch, seq)
        >>> logits = model.forward(tokens)
        >>> assert logits.shape == (2, 10, 1000)  # (batch, seq, vocab)
        """

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Embedding layer
        self.embedding_layer = Embeddinglayer(vocab_size, embed_dim, max_seq_len)

        # Stack of transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads)
            self.blocks.append(block)

        # Final layer normalization
        self.ln_f = LayerNorm(embed_dim)

        # Language modeling head (projects to vocabulary)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)


    def forward(self, tokens):
        """
        Forward pass through GPT model.

        COMPUTATION FLOW:
        tokens -> embed -> pos_embed -> blocks -> ln_f -> lm_head -> logits

        CAUSAL MASKING:
        For autoregressive generation, we need to prevent tokens from
        seeing future tokens. This is handled by the attention mask.
        """

        batch_size, seq_len = tokens.shape

        # Pass tokens to embedding layer to get token embeddings and positional embeddings
        x = self.embedding_layer.forward(tokens)

        # Create causal mask for autoregressive generation
        mask = self._create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer normalization
        x = self.ln_f.forward(x)

        # Language modeling head
        logits = self.lm_head.forward(x)

        return logits

    def __call__(self, tokens):
        """Allows the GPT model to be called like a function."""
        return self.forward(tokens)

    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions."""
        mask = np.tril(np.ones((seq_len, seq_len)))
        return Tensor(mask)

    def _sample_next_token(self, logits, temperature=1.0):
        """
        Sample one token from vocabulary logits using temperature scaling.

        EXAMPLE:
        >>> logits = np.array([[1.0, 2.0, 3.0]])  # Raw model output
        >>> token = model._sample_next_token(logits, temperature=1.0)
        >>> assert 0 <= token < 3  # Valid token index
        """
        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Convert to probabilities (softmax with numerical stability)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Sample next token from probability distribution
        next_token = rng.choice(self.vocab_size, p=probs[0])
        return next_token

    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """
        Generate text autoregressively by repeatedly sampling next tokens.

        EXAMPLE:
        >>> model = GPT(vocab_size=100, embed_dim=64, num_layers=2, num_heads=4)
        >>> prompt = Tensor([[1, 2, 3]])  # Some token sequence
        >>> generated = model.generate(prompt, max_new_tokens=5)
        >>> assert generated.shape[1] == 3 + 5  # original + new tokens
        """

        current_tokens = Tensor(prompt_tokens.data.copy())

        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self.forward(current_tokens)

            # Get logits for last position (next token prediction)
            last_logits = logits.data[:, -1, :]  # (batch_size, vocab_size)

            # Sample next token using helper
            next_token_id = self._sample_next_token(last_logits, temperature)

            # Append to sequence
            next_token = np.array([[next_token_id]])
            current_tokens = Tensor(np.concatenate([current_tokens.data, next_token], axis=1))

        return current_tokens

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.embedding_layer.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())

        return params

    
        