"""
## Systems Analysis: Training Performance and Memory

Training systems have significant resource requirements. Understanding memory usage, checkpoint sizes, and training overhead helps optimize production ML pipelines.

### Training Memory Breakdown

```
Training Memory Requirements:

Forward Pass Memory:
┌─────────────────┐
│ Activations     │ ← Stored for backward pass
├─────────────────┤
│ Model Params    │ ← Network weights
└─────────────────┘

Backward Pass Memory:
┌─────────────────┐
│ Gradients       │ ← Same size as params
├─────────────────┤
│ Optimizer State │ ← 2-3× params (momentum, Adam buffers)
└─────────────────┘

Checkpoint Memory:
┌─────────────────┐
│ Model State     │ ← Full parameter snapshot
├─────────────────┤
│ Optimizer State │ ← All momentum/Adam buffers
├─────────────────┤
│ Training Meta   │ ← Epoch, history, scheduler
└─────────────────┘

Total Training Memory ≈ 4-6× Model Parameters
  (4× covers params + grads + Adam moments; 5-6× when activation memory is included)
```

### Key Systems Insights

**Gradient Accumulation Trade-off:**
- Effective batch size = accumulation_steps × actual_batch_size
- Memory: Fixed (only 1 batch in memory at a time)
- Time: Increases linearly with accumulation steps
- Use case: Large models that don't fit with desired batch size

**Checkpoint Size:**
- Base model: 1× parameters
- With optimizer (Adam): ~3× parameters
- With full history: Additional metadata
- Compression: Pickle overhead ~10-20%
"""

import numpy as np
rng = np.random.default_rng(7)
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import sys
import os

from core.tensor import Tensor
from core.layers import Linear
from core.losses import MSELoss, CrossEntropyLoss
from core.optimizers import SGD, AdamW

from core.autograd import enable_autograd
enable_autograd()

# Constants for learning rate scheduling defaults
DEFAULT_MAX_LR = 0.1  # Default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01  # Default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS = 100  # Default total epochs for learning rate schedule


class CosineSchedule:
    """
    Cosine annealing learning rate schedule.

    Starts at max_lr, decreases following a cosine curve to min_lr over T epochs.
    This provides aggressive learning initially, then fine-tuning at the end.

    EXAMPLE:
    >>> schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
    >>> print(schedule.get_lr(0))    # Start: 0.1
    >>> print(schedule.get_lr(50))   # Middle: ~0.055
    >>> print(schedule.get_lr(100))  # End: 0.01
    """

    def __init__(
        self,
        max_lr: float = DEFAULT_MAX_LR,
        min_lr: float = DEFAULT_MIN_LR,
        total_epochs: int = DEFAULT_TOTAL_EPOCHS,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        # Without it, cos keeps going past π and the LR would climb toward max_lr again
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor



def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    """
    Clip gradients by global norm to prevent exploding gradients.

    This is crucial for training stability, especially with RNNs and DNN.
    Instead of clipping each gradient individually, we compute the global norm
    across all parameters and scale uniformly if needed.

    EXAMPLE:
    >>> params = [Tensor([1, 2, 3], requires_grad=True)]
    >>> params[0].grad = Tensor([10, 20, 30])  # Large gradients
    >>> original_norm = clip_grad_norm(params, max_norm=1.0)
    >>> print(f"Clipped norm: {np.linalg.norm(params[0].grad.data):.2f}")  # Should be ≤ 1.0
    """
    if not parameters:
        return 0.0

    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            # Handle both Tensor gradients and numpy array gradients
            if isinstance(param.grad, np.ndarray):
                grad_data = param.grad
            else:
                # Trust that Tensor has .data attribute
                grad_data = param.grad.data
            total_norm += np.sum(grad_data ** 2)

    total_norm = np.sqrt(total_norm)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                # Handle both Tensor gradients and numpy array gradients
                if isinstance(param.grad, np.ndarray):
                    param.grad = param.grad * clip_coef
                else:
                    # Trust that Tensor has .data attribute
                    param.grad.data = param.grad.data * clip_coef
    return float(total_norm)


class Trainer:
    """
    Complete training orchestrator for neural networks.

    Handles the full training lifecycle: forward pass, loss compucation,
    backward pass, optimization, scheduling, checkpointing, and evaluation.

    This is the central class that brings together all the components
    you've built in previous modules.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        grad_clip_norm=None,
    ):
        """
        Initialize trainer with model and training components.

        Args:
            model: Neural network to train (must have .forward and .parameters())
            optimizer: Parameter update strategy (SGD, Adam, etc.)
            loss_fn: Loss function (CrossEntropy, MSE, etc.)
            scheduler: Optional learning rate scheduler (e.g., CosineSchedule)
            grad_clip_norm: Optional max gradient norm for clipping (float)
        """
        self.model = model
        self.optimizer = optimizer 
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

        # Enable gradient tracking for all model parameters.
        # Layers (e.g., Linear) may be created without requires_grad=True,
        # so we set it explicitely here to ensure backward() populates param.grad.
        # Guard against raw numpy arrays passed by test stubs or non-Tensor params.
        for param in model.parameters():
            if isinstance(param, Tensor):
                param.requires_grad = True

        # Training state
        self.epoch = 0
        self.step = 0
        self.training_mode = True

        # History tracking
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': [],
        }

    # Training loop flow:
    # for each batch in dataloader:
    #   outputs = model.forward(inputs)
    #   loss = loss_fn(outputs, targets)
    #   loss.backward()
    #   optimizer.step()
    #   optimizer.zero_grad()
    # scheduler.get_lr(epoch)
    # ----------------------------------------
    # With gradient accumulation, the update step happens every N batches instead
    # of every batch, enabling larger effective batch sizes without more memory.
    # ----------------------------------------
    # We'll build this in three pieces:
    #   1. Process a single batch
    #   2. Perform an optimizer update
    #   3. Compose them into the full epoch loop

    # STEP 1: Process a single batch
    #   The inner loop body: run forward, compute loss, and run backward pass
    #   with scaled gradients for accumulation.

    def _process_batch(self, inputs, targets, accumulation_steps):
        """
        Process one batch: forward pass, loss computation, backward pass.

        Args:
            inputs: Input tensor for this batch
            targets: Target tensor for this batch
            accumulation_steps: Number of batches per optimizer update (for scaling)
        """
        # Forward pass
        outputs = self.model.forward(inputs)
        loss = self.loss_fn.forward(outputs, targets)

        # Scale loss for accumulation
        scaled_loss = loss.data / accumulation_steps

        # Backward pass with scaled gradient
        scaled_gradient = np.ones_like(loss.data) / accumulation_steps
        loss.backward(scaled_gradient)

        return float(scaled_loss)

    # STEP 2: Perform Optimizer update
    #   When enough gradients have accumulated, clip them (if configured),
    #   step the optimizer, and reset gradients for the next accumulation window.

    def _optimizer_update(self):
        """Clip gradients (if enabled) and step the optimizer."""

        if self.grad_clip_norm is not None:
            params = self.model.parameters()
            clip_grad_norm(params, self.grad_clip_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

    # STEP 3: Compose the full epoch loop
    def train_epoch(self, dataloader, accumulation_steps=1):
        """
        Train for one epoch through the dataset.

        Args:
            dataloader: Iterable yielding (inputs, targets) batches
            accumulation_steps: Number of batches to accumulate before update

        Returns:
            Average loss for the epoch (float)
        """
        self.model.training = True
        self.training_mode = True
        
        # Update scheduler at the start of each epoch so LR is set before training begins.
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            self.optimizer.lr = current_lr
            self.history['learning_rates'].append(current_lr)

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            accumulated_loss += self._process_batch(inputs, targets, accumulation_steps)

            # Update parameters every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self._optimizer_update()
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                self.step += 1

        # Handle remaining accumulated gradients
        if accumulated_loss > 0:
            self._optimizer_update()
            total_loss += accumulated_loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)

        self.epoch += 1
        return avg_loss

    # ========================================================================
    # Trainer.evaluate - Measuring Model Performance

    def evaluate(self, dataloader):
        """
        Evaluation runs the model in inference mode: forward pass only, no gradient
        updates. This tells you how well the model generalizes to data it hasn't
        trained on.

        Evaluation flow:
            model.training = False
            for each batch in dataloader:
                outputs = model.forward(inputs)
                loss = loss_fn(outputs, targets)
                accumulate loss + accuracy
            return avg_loss, accuracy

        Key difference from training: no backward pass, no optimizer step,
        no gradient clipping.

        Args:
            dataloader: Iterable yielding (inputs, targets) batches

        Returns:
            Tuple of (average_loss, accuracy)
        """

        self.model.training = False
        self.training_mode = False

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for inputs, targets in dataloader:
            # Forward pass only
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            total_loss += loss.data
            num_batches += 1

            # Calculate accuracy (for classification only).
            # outputs.data.shape[-1] > 1 distinguishes true multi-class (C logits)
            # from regression with a single output neuron (shape (N,1)), which would
            # otherwise enter this branch and produce argmax=0 for every sample.
            if len(outputs.data.shape) > 1 and outputs.data.shape[-1] > 1:
                predictions = np.argmax(outputs.data, axis=1)
                if len(targets.data.shape) == 1:
                    correct += np.sum(predictions == targets.data)
                else:
                    correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                total += len(predictions)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history['eval_loss'].append(avg_loss)

        return avg_loss, accuracy

    def save_checkpoint(self, path: str):
        """
        Save complete training state for resumption.
    
        Args:
            path: File path to save checkpoint (.pkl)

        EXAMPLE:
        >>> eval_loss, accuracy = trainer.evaluate(test_data)
        >>> print(f"Eval loss: {eval_loss:.4f}, Accuracy: {accuracy:.2%}")
        """

        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state': self._get_model_state(),
            'optimizer_state': self._get_optimizer_state(),
            'scheduler_state': self._get_scheduler_state(),
            'history': self.history,
            'training_mode': self.training_mode,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file first, then atomically replace the target.
        # Writing directly to `path` would leave a partially-written (corrupt)
        # checkpoint in place if the process is killed, the disk fills up, or
        # pickling fails partway through, silently destroying whatever good
        # checkpoint was there before. os.replace() is atomic on both POSIX
        # and Windows: `path` is always either the old complete checkpoint or
        # the new one, never a partial write.
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        
    def load_checkpoint(self, path: str):
        """
        Load training state from checkpoint.
    
        Args:
            path: File path to load checkpoint from (.pkl)

        EXAMPLE:
        >>> trainer.save_checkpoint('/tmp/checkpoint.pkl')
        >>> trainer.epoch = 999  # Some change
        >>> trainer.load_checkpoint('/tmp/checkpoint.pkl')
        >>> print(trainer.epoch)  # Restored to original value
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
    
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        self.training_mode = checkpoint['training_mode']
    
        # Restore states
        if 'model_state' in checkpoint:
            self._set_model_state(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self._set_optimizer_state(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self._set_scheduler_state(checkpoint['scheduler_state'])

        
    
            

    # ── Private serialization helpers
    def _get_model_state(self):
        """Extract model parameters for checkpointing."""
        return {i: param.data.copy() for i, param in enumerate(self.model.parameters())}

    def _set_model_state(self, state):
        """Restore model parameters from checkpoint."""
        for i, param in enumerate(self.model.parameters()):
            if i in state:
                param.data = state[i].copy()

    def _get_optimizer_state(self):
        """Extract optimizer state for checkpointing."""
        state = {}
        state['lr'] = self.optimizer.lr
        if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
            momentum_state = self.optimizer.get_momentum_state()
            if momentum_state is not None:
                state['momentum_buffers'] = momentum_state
        return state

    def _set_optimizer_state(self, state):
        """Restore optimizer state from checkpoint."""
        if 'lr' in state:
            self.optimizer.lr = state['lr']
        if 'momentum_buffers' in state:
            if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
                self.optimizer.set_momentum_state(state['momentum_buffers'])

    def _get_scheduler_state(self):
        """Extract scheduler state for checkpointing."""
        if self.scheduler is None:
            return None
        return {
            'max_lr': getattr(self.scheduler, 'max_lr', None),
            'min_lr': getattr(self.scheduler, 'min_lr', None),
            'total_epochs': getattr(self.scheduler, 'total_epochs', None)
        }

    def _set_scheduler_state(self, state):
        """Restore scheduler state from checkpoint."""
        if state is None or self.scheduler is None:
            return
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)
