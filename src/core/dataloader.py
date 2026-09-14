"""
## Understanding the Data Pipeline

Before we implement anything, let's understand what happens when neural networks "eat" data.
The journey from raw data to trained models follows a specific pipeline that every ML engineer must master.

### The Data Pipeline Journey

Imagine you have 50,000 images of cats and dogs, and you want to train a neural network to classify them:

```
Raw Data Storage          Dataset Interface         DataLoader Batching         Model Input
┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐      ┌─────────────┐
│ cat_001.jpg     │      │ dataset[0]       │      │ Batch 1:           │      │ model(batch)│
│ dog_023.jpg     │ ───> │ dataset[1]       │ ───> │ [cat, dog, cat]    │ ───> │ compute     │
│ cat_045.jpg     │      │ dataset[2]       │      │ Batch 2:           │      │ loss        │
│ ...             │      │ ...              │      │ [dog, cat, dog]    │      │ repeat      │
│ (50,000 files)  │      │ dataset[49999]   │      │ ...                │      │             │
└─────────────────┘      └──────────────────┘      └────────────────────┘      └─────────────┘
```

### Why This Pipeline Matters

**Individual Access (Dataset)**: Neural networks can't process 50,000 files at once. We need a way to access 
one sample at a time: "Give me image #1,247".

**Batch Processing (DataLoader)**: GPUs are parallel machines - they're much faster processing 32 images simultaneously than 1 image 32 times.

**Memory Efficiency**: Loading all 50,000 images into memory would require ~150GB. Instead, we load only the current batch (~150MB).

**Training Variety**: Shuffling ensures the model sees different combinations each epoch, preventing memorization.
"""


import random
import sys
import time
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import numpy as np
rng = np.random.default_rng(7)

from core.tensor import Tensor



class Dataset(ABC):
    """
    Abtract base class for all datasets.

    Provides the fundamental interface that all datasets must implement:
        - __len__(): Returns the total number of samples
        - __getitem__(idx): Return the sample at given index
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        This method must be implemented by all subclasses to enable
        len(dataset) calls and batch size calculations.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """
        Return the sample at the given index.

        Args:
            idx: Index of the sample to retrieve (0 <= idx < len(dataset))

        Returns:
            The sample at index idx. Format depends on the dataset implementation.
            Could be (data, label) tuple, single tensor, etc.
        """
        pass


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors for supervised training.

    Each sample is a tuple of tensors from the same index across all input tensors.
    All tensors must have the same size in their first dimension.

    ```
    Input Tensors (aligned by first dimension):
      Features Tensor        Labels Tensor         Metadata Tensor
      ┌─────────────────┐   ┌───────────────┐     ┌─────────────────┐
      │ [1.2, 3.4, 5.6] │   │ 0 (cat)       │     │ "image_001.jpg" │ ← Sample 0
      │ [2.1, 4.3, 6.5] │   │ 1 (dog)       │     │ "image_002.jpg" │ ← Sample 1
      │ [3.0, 5.2, 7.4] │   │ 0 (cat)       │     │ "image_003.jpg" │ ← Sample 2
      │ ...             │   │ ...           │     │ ...             │
      └─────────────────┘   └───────────────┘     └─────────────────┘
            (N, 3)               (N,)                   (N,)
    
    Dataset Access:
      dataset[1] → (Tensor([2.1, 4.3, 6.5]), Tensor(1), "image_002.jpg")
    ```
    """
    def __init__(self, *tensors):
        """
        Create dataset from multiple tensors.

        Args:
            *tensors: Variable number of Tensor objects.

        All tensors must have the same size in their first dimension.
        """
        assert len(tensors) > 0, "Must provide at least one tensor"

        # Store all tensors
        self.tensors = tensors

        # Validate all tensors have same first dimension
        first_size = len(tensors[0].data)
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f"Tensor size mismatch in TensorDataset\n"
                    f"Tensor 0 has {first_size} samples, but tensor {i} has {len(tensor.data)} samples"
                )

    def __len__(self) -> int:
        """
        Return number of samples (size of first dimension).

        EXAMPLES:
        >>> features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples
        >>> labels = Tensor([0, 1, 0])
        >>> dataset = TensorDataset(feature, labels)
        >>> print(len(dataset))  # 3
        """
        return len(self.tensors[0].data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """
        Return tuple of tensor slices at given index.

        EXAMPLES:
        >>> features = Tensor([[1, 2], [3, 4], [5, 6]])
        >>> labels = Tensor([0, 1, 0])
        >>> dataset = TensorDataset(features, labels)
        >>> sample = dataset[1]
        >>> # Returns: (Tensor([3, 4]), Tensor(1))
        >>> last = dataset[-1]
        >>> Returns: (Tensor([5, 6]), Tensor(0))
        """
        
        original_idx = idx
        if idx < 0:
            idx += len(self)

        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {original_idx} out of range for dataset of size {len(self)}")

        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)


class DataLoader:
    """
    Data loader with batching and shuffling support.

    Wraps a dataset to provide batched iteration with optional shuffling.
    Essential for efficient training with mini-batch gradient descent.

    EXAMPLE;
    >>> dataset = TensorDataset(Tensor([[1, 2], [3, 4], [5, 6]]), Tensor([0, 1, 0]))
    >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
    >>> for batch in loader:
    ...     features_batch, labels_batch = batch
    ...     print(f"Features: {features_batch.shape}, Labels: {labels_batch.shape}")
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        """
        Create DataLoader for batched iteration.

        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """
        Return number of batches per epoch.
        EXAMPLES:
        >>> dataset = TensorDataset(Tensor([[1], [2], [3], [4], [5]]))
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> print(len(loader))  # 3 (batches: [2, 2, 1])
        """
        # Calculate number of complete batches
        # If we calculate the batches per epoch by N//B, it'd ignore the final incomplete batch
        # EXAMPLE: N=100, B=8 --> N//B=12 with 4 samples remained in the final incomplete batch.
        # By (N + B - 1) // B, we keep the remaining 4 as a separete batch, as it should be!
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        """
        Return iterator over batches.

        EXAMPLE:
        >>> dataset = TensorDataset(Tensor([[1], [2], [3], [4]]))
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> for batch in loader:
        ...     print(batch[0].shape)  # (2, 1)
        """
        # Create list of indices
        indices = list(range(len(self.dataset)))

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i: i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # Collate batch - convert list of tuples to tuple of tensors
            yield self._collate_batch(batch)

    def _collate_batch(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        """
        Collate individual samples into batch tensors.

        Args:
            batch: List of sample tuples from dataset

        Returns:
            Tuple of batched tensors

        EXAMPLE:
        >>> # batch = [(Tensor([1,2]), Tensor(0)),
        ...            (Tensor([3,4]), Tensor(1))]
        >>> # Returns: (Tensor([[1,2], [3,4]]), Tensor([0, 1]))
        """

        if len(batch) == 0:
            return ()

        # Determine number of tensors per sample
        num_tensors = len(batch[0])

        # Group tensors by position
        batched_tensors = []
        for tensor_idx in range(num_tensors):
            # Extract all tensors at this position
            tensor_list = [sample[tensor_idx].data for sample in batch]

            # Stack into batch tensor
            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))

        return tuple(batched_tensors)


class RandomHorizontalFlip:
    """
    Randomly flip images horizontally with given probability.

    A simple but effective augmentation for most image datasets.
    Flipping is appropriate when horizontal orientation doesn't change class
    (cats, dogs, cars, but not digits or texts)

    Args:
        p: probability of flipping (default: 0.5)
    """

    def __init__(self, p=0.5):
        """Initialize RandomHorizontalFlip"""
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"Invalid flip probability: {p}\n"
                f"p must be between 0.0 and 1.0"
            )
        self.p = p

    def __call__(self, x):
        """Apply random horizontal flip to input."""
        if np.random.random() < self.p:
            is_tensor = isinstance(x, Tensor)
            data = x.data if is_tensor else x

            # Determine width axis for HW/CHW/HWC
            # Convention (matching _pad_image with RandomCrop): check shape[0]
            # for channels-first (C, H, W) where C <= 4. This is the standard
            # PyTorch NCHW convention for 3D image tensors.
            if data.ndim == 2:
                # (H, W)
                axis = -1
            elif data.ndim == 3:
                if data.shape[0] <= 4:
                    # Channel-first: (C, H, W) -> flip width (last axis)
                    axis = -1
                else:
                    # Channels-last (H, W, C) -> flip width (second-to-last)
                    axis = -2
            else:
                raise ValueError(
                    f"RandomHorizontalFlip requires at least 2D input\n"
                    f"Got {data.ndim}D input with shape {data.shape}."
                )

            flipped = np.flip(data, axis=axis).copy()
            return Tensor(flipped) if is_tensor else flipped
        return x


def _pad_image(data, padding):
    """
    Detect image format and apply zero-padding to spatial dimensions only.

    EXAMPLE:
    >>> img = np.ones((3, 4, 4))  # (C, H, W) = (3, 4, 4)
    >>> padded = _pad_image(img, padding=2)
    >>> print(padded.shape)  # (3, 8, 8) — channels unchanged, spatial +4
    """
    if data.ndim == 2:
        # (H, W) format -> Pad both axis
        return np.pad(data, padding, mode='constant', constant_values=0)
    elif data.ndim == 3:
        if data.shape[0] <= 4:
            # Channels-first: (C, H, W) -> pad only H and W
            return np.pad(data, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
        else:
            # Channels-lsat: (H, W, C) -> pad only H and W
            return np.pad(data, ((padding, padding), (padding, padding), (0, 0)), mode="constant", constant_values=0)
    else:
        raise ValueError(
            f"RandomCrop requires 2D or 3D input\n"
            f"Got {data.ndim}D input with shape {data.shape}\n"
            f"Expected formats: (H, W) for grayscale, (C, H, W) or (H, W, C) for color images\n"
        )


def _random_crop_region(padded_h, padded_w, target_h, target_w):
    """
    Sample a random (top, left) position for cropping.

    EXAMPLE:
    >>> top, left = _random_crop_region(40, 40, 32, 32)
    >>> # top in [0, 8], left in [0, 8]    
    """
    top = rng.integers(0, padded_h - target_h + 1)
    left = rng.integers(0, padded_w - target_w + 1)
    return top, left


class RandomCrop:
    """
    Randomly crop image after padding.

    This is the standard augmentation for CIFAR-10:
    1. Pad image by `padding` pixels on each side
    2. Randomly crop back to original size

    This simulates small translations in the image, forcing the model
    to recognize objects regardless of their exact position.

    Args:
        size: Output crop size (int for square, or tuple (H, W))
        padding: Pixels to pad on each side before cropping (default: 4)
    """
    def __init__(self, size, padding=4):
        """
        Initialize RandomCrop.

        EXAMPLE:
        >>> crop = RandomCrop(32, padding=4)  # CIFAR-10 standard
        >>> # Pads to 40x40, then crops back to 32x32
        """

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, x):
        """
        Apply random crop after padding.

        Composes _pad_image and _random_crop_region to perform:
        1. Pad the image with zeros on spatial dimensions
        2. Sample a random crop position
        3. Extract the crop and return

        EXAMPLE:
        >>> crop = RandomCrop(32, padding=4)
        >>> img = rng.standard_normal((3, 32, 32))
        >>> out = crop(img)
        >>> print(out.shape)  # (3, 32, 32)        
        """
        is_tensor = isinstance(x, Tensor)
        data = x.data if is_tensor else x

        target_h, target_w = self.size

        # STEP 1: Pad the image (handles format detection internally)
        padded = _pad_image(data, self.padding)

        # STEP 2: Determine padded spatial dims and sample crop position
        if data.ndim == 2:
            padded_h, padded_w = padded.shape
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[top:top + target_h, left:left + target_w]
        elif data.shape[0] <= 4:
            # Channels-first: (C, H, W)
            padded_h, padded_w = padded.shape[1], padded.shape[2]
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[:, top:top + target_h, left:left + target_w]
        else:
            # Channels-last: (H, W, C)
            padded_h, padded_w = padded.shape[0], padded.shape[1]
            top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
            cropped = padded[top:top + target_h, left:left + target_w, :]

        return Tensor(cropped) if is_tensor else cropped


class Compose:
    """
    Compose multiple transforms into a pipeline.

    Applies transforms in sequence, passing output of each
    as input to the next.

    Args:
        transforms: List of transform callables.
    """
    def __init__(self, transforms):
        """
        Initialize Compose with list of transforms.

        EXAMPLE:
        >>> transforms = Compose([
        ...     RandomHorizontalFlip(0.5),
        ...     RandomCrop(32, padding=4)
        ... ])
        """
        self.transforms = transforms

    def __call__(self, x):
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            x = transform(x)
        return x
