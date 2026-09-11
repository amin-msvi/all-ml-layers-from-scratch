"""
### Network Architecture Visualization
```
MNIST Classification Network (3-Layer MLP):

    Input Layer          Hidden Layer 1        Hidden Layer 2        Output Layer
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     784         │    │      256        │    │      128        │    │       10        │
│   Pixels        │───▶│   Features      │───▶│   Features      │───▶│    Classes      │
│  (28×28 image)  │    │   + ReLU        │    │   + ReLU        │    │  (0-9 digits)   │
│                 │    │   + Dropout     │    │   + Dropout     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        ↓                       ↓                       ↓                       ↓
   "Raw pixels"          "First hidden features"        "Second hidden features"        "Output predictions"

Data Flow:
[32, 784] → Linear(784,256) → ReLU → Dropout(0.5) → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,10) → [32, 10]
```

### Parameter Count Analysis
```
Parameter Breakdown (Manual Layer Composition):
┌─────────────────────────────────────────────────────────────┐
│ layer1 = Linear(784 → 256)                                  │
│   Weights: 784 × 256 = 200,704 params                       │
│   Bias:    256 params                                       │
│   Subtotal: 200,960 params                                  │
├─────────────────────────────────────────────────────────────┤
│ activation1 = ReLU(), dropout1 = Dropout(0.5)               │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ layer2 = Linear(256 → 128)                                  │
│   Weights: 256 × 128 = 32,768 params                        │
│   Bias:    128 params                                       │
│   Subtotal: 32,896 params                                   │
├─────────────────────────────────────────────────────────────┤
│ activation2 = ReLU(), dropout2 = Dropout(0.3)               │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ layer3 = Linear(128 → 10)                                   │
│   Weights: 128 × 10 = 1,280 params                          │
│   Bias:    10 params                                        │
│   Subtotal: 1,290 params                                    │
└─────────────────────────────────────────────────────────────┘
                    TOTAL: 235,146 parameters
                    Memory: ~940 KB (float32)
```

## Systems Analysis: Memory and Performance

Now let's analyze the systems characteristics of our layer implementations. Understanding memory usage and computational complexity helps us build efficient neural networks.

### Memory Analysis Overview
```
Layer Memory Components:
┌─────────────────────────────────────────────────────────────┐
│                    PARAMETER MEMORY                         │
├─────────────────────────────────────────────────────────────┤
│ • Weights: Persistent, shared across batches                │
│ • Biases: Small but necessary for output shifting           │
│ • Total: Grows with network width and depth                 │
├─────────────────────────────────────────────────────────────┤
│                   ACTIVATION MEMORY                         │
├─────────────────────────────────────────────────────────────┤
│ • Input tensors: batch_size × features × 4 bytes            │
│ • Output tensors: batch_size × features × 4 bytes           │
│ • Intermediate results during forward pass                  │
│ • Total: Grows with batch size and layer width              │
├─────────────────────────────────────────────────────────────┤
│                   TEMPORARY MEMORY                          │
├─────────────────────────────────────────────────────────────┤
│ • Dropout masks: batch_size × features × 1 byte             │
│ • Computation buffers for matrix operations                 │
│ • Total: Peak during forward/backward passes                │
└─────────────────────────────────────────────────────────────┘
```

### Computational Complexity Overview
```
Layer Operation Complexity:
┌─────────────────────────────────────────────────────────────┐
│ Linear Layer Forward Pass:                                  │
│   Matrix Multiply: O(batch × in_features × out_features)    │
│   Bias Addition: O(batch × out_features)                    │
│   Dominant: Matrix multiplication                           │
├─────────────────────────────────────────────────────────────┤
│ Multi-layer Forward Pass:                                   │
│   Sum of all layer complexities                             │
│   Memory: Peak of all intermediate activations              │
├─────────────────────────────────────────────────────────────┤
│ Dropout Forward Pass:                                       │
│   Mask Generation: O(elements)                              │
│   Element-wise Multiply: O(elements)                        │
│   Overhead: Minimal compared to linear layers               │
└─────────────────────────────────────────────────────────────┘
```
"""


import numpy as np
# Module-level RNG is seeded so Linear weight init is deterministic by default.
# This is what the integration test suite (and any cross-run reproducibility)
# relies on. Demo scripts that want fresh weights every run rebind this name
# to an unseeded RNG locally before constructing their model — see
# milestones/01_1958_perceptron/01_rosenblatt_forward.py for the pattern.
rng = np.random.default_rng(7)

from core.tensor import Tensor
from core.activations import ReLU, Sigmoid

# Constants for weight initialization
# Note: True Xavier/Glorot uses sqrt(2/(fan_in+fan_out)), but we use the simpler
# LeCun-style sqrt(1/fan_in) for pedagogical clarity. Both achieve stable gradients.
INIT_SCALE_FACTOR = 1.0  # LeCun-style initialization: sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU

# Constants for dropout
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)


class Layer:
    """
    Base class for all neural network layers.

    All layers should inherit from this class and implement:
        - forward(x): Compute layer output
        - parameters(): Return list of trainable parameters

    The __call__ method is provided to make layers callable.
    """

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x: Input tensor

        Returns:
            Output  tensor after transformation
        """
        raise NotImplementedError(
            f"forward() not implemented in {self.__class__.__name__}\n"
            f"  The Layer base class requires subclasses to implement forward()\n"
            f"  forward() defines how input data is transformed by this layer\n"
            f"  Add this method to your class:\n"
            f"     def forward(self, x):\n"
            f"         # Your transformation logic here\n"
            f"         return transformed_x"
        )

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        """
        Return list of trainable parameters.

        Returns:
            List of Tensor objects (weights and biases)
        """
        return []  # Base class has no parameters

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Linear(Layer):
    """
    Linear (fully connected) layer: y = xW + b

    This is the fundamental building block of neural networks.
    Applies a linear transformation to incoming data.
    """

    def __init__(self, in_features, out_features, bias=True):
        """Initialize linear layer with proper weight initialization."""
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        weight_data = rng.standard_normal((in_features, out_features)) * scale
        self.weight = Tensor(weight_data)

        # Initialize bias to zeros or None
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        # Linear transformation: y = xW
        output = x.matmul(self.weight)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str}"


class Dropout(Layer):
    """
    Dropout layer for regularization.

    Druing training: randomly zeros elements with probability p, scales survivors by 1/(1-p)
    During inference: passes input through unchanged

    This prevents overfitting by forcing the network to not rely on specific neurons.
    """

    def __init__(self, p=0.5):
        """Initiate dropout layer"""
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(
                f"Invalid dropout probability: {p}\n"
                f"  p must be between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}\n"
                f"  p is the probability of DROPPING a neuron (not keeping it!)\n"
                f"  p=0.0 means keep all neurons (no dropout)\n"
                f"  p=0.5 means drop 50% of neurons randomly\n"
                f"  p=1.0 means drop all neurons (zero output)\n"
                f"  Common values: Dropout(0.1) for light, Dropout(0.3) for moderate, Dropout(0.5) for aggressive"
            )
        self.p = p

    def _should_apply_dropout(self, training):
        """
        Determine whether dropout should be applied.

        Dropout is a training-time technique. During inference the full
        network is used, so dropout is skipped. It is also skipped when p=0
        (no neurons are dropped) since the result would be the identity.
        """
        return training and self.p > DROPOUT_MIN_PROB

    def _generate_dropout_mask(self, shape):
        """
        Generate a random dropout mask with inverted scaling.

        The mask has the same shape as the input. Each element is either 
        0 (dropped) or 1/(1-p) (kept and scaled). Scaling at training time
        keeps the expected value of each element unchanged, so no adjustment
        is needed at inference. This trick is called "inverted dropout".

        ```
        Example with p=0.5 (keep_prob=0.5, scale=2.0):
        random draw:  [0.3,  0.8,  0.1,  0.6]
                        ↓     ↓     ↓     ↓
        keep?         [yes,  no,  yes,  no ]   (< 0.5?)
                        ↓     ↓     ↓     ↓
        mask:         [2.0,  0.0,  2.0,  0.0]  (kept × scale, dropped × 0)
        ```
        """
        keep_prob = 1.0 - self.p
        binary_mask = (rng.random(shape) < keep_prob).astype(np.float32)
        scale = 1.0 / keep_prob
        return Tensor(binary_mask * scale)

    def forward(self, x, training=True):
        """
        Forward pass through dropout layer.

        Composes the two helpers: first decide whether dropout applied,
        then generate and apply the mask if it does.
        """

        if not self._should_apply_dropout(training):
            return x

        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data))

        mask = self._generate_dropout_mask(x.data.shape)
        return x * mask

    def __call__(self, x, training=True):
        return self.forward(x, training)

    def parameters(self):
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential:
    """
    Container that chains layers together sequentially.

    After you understand explicit layer composition, Sequential provides
    a convenient way to bundle layers together.

    Example:
        >>> model = Sequential(
        ...     Linear(784, 128),
        ...     ReLU(),
        ...     Linear(128, 10)
        ... )
        >>> output = model(input_tensor)
        >>> params = model.parameters()  # All parameters from all layers
    """

    def __init__(self, *layers):
        """Initialize with layers to chain together."""
        # Accept both Sequential(layer1, layer2) and Sequencial([layer1, layer2])
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self, x, training=True):
        """Forward pass through all layers sequentially.

        Passes training=True/False to layers that support it (e.g. Dropout),
        and falls back to a plain forward(x) call for layers that don't.
        This  lets you switch between training and eval mode with one flag:
            output = model.forward(x, training=False)  # eval: Dropout disabled
            output = model.forward(x, training=True)   # train: Dropout active
        """
        for layer in self.layers:
            try:
                x = layer.forward(x, training=training)
            except TypeError:
                x = layer.forward(x)
        return x

    def __call__(self, x, training=True):
        """Allow model to be called like a function."""
        return self.forward(x, training=training)

    def parameters(self):
        """Collect all parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequencial({layer_reprs})"
