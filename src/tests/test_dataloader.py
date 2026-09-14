import numpy as np

from core.dataloader import (
    Dataset,
    TensorDataset,
    DataLoader,
    _pad_image,
    _random_crop_region,
    RandomHorizontalFlip,
    RandomCrop,
    Compose,
    rng
)
from core.tensor import Tensor

def test_unit_dataset():
    """Test Dataset abstract base class."""
    print("Unit Test: Dataset Abstract Base Class...")

    # Test that Dataset is properly abstract
    try:
        dataset = Dataset()
        assert False, "Should not be able to instantiate abstract Dataset"
    except TypeError:
        print("Dataset is properly abstract")

    # Test concrete implementation
    class TestDataset(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return f"item_{idx}"

    dataset = TestDataset(10)
    assert len(dataset) == 10
    assert dataset[0] == "item_0"
    assert dataset[9] == "item_9"

    print("Dataset interface works correctly!")


def test_unit_tensordataset():
    """Test TensorDataset implementation."""
    print("Unit Test: TensorDataset...")

    # Test basic functionality
    features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
    labels = Tensor([0, 1, 0])                   # 3 labels

    dataset = TensorDataset(features, labels)

    # Test length
    assert len(dataset) == 3, f"Expected length 3, got {len(dataset)}"

    # Test indexing
    sample = dataset[0]
    assert len(sample) == 2, "Should return tuple with 2 tensors"
    assert np.array_equal(sample[0].data, [1, 2]), f"Wrong features: {sample[0].data}"
    assert sample[1].data == 0, f"Wrong label: {sample[1].data}"

    sample = dataset[1]
    assert np.array_equal(sample[1].data, 1), f"Wrong label at index 1: {sample[1].data}"

    # Test error handling
    try:
        dataset[10]  # Out of bounds
        assert False, "Should raise IndexError for out of bounds access"
    except IndexError:
        pass

    # Test mismatched tensor sizes
    try:
        bad_features = Tensor([[1, 2], [3, 4]])  # Only 2 samples
        bad_labels = Tensor([0, 1, 0])           # 3 labels - mismatch!
        TensorDataset(bad_features, bad_labels)
        assert False, "Should raise error for mismatched tensor sizes"
    except ValueError:
        pass

    print("TensorDataset works correctly!")


def test_unit_pad_image():
    """Test _pad_image helper."""
    print("Unit Test: _pad_image...")

    # Test 2D (H, W) — grayscale
    img_hw = np.ones((8, 8))
    padded_hw = _pad_image(img_hw, padding=4)
    assert padded_hw.shape == (16, 16), f"2D pad shape wrong: {padded_hw.shape}"
    # Corners should be zero (padding), center should be ones
    assert padded_hw[0, 0] == 0.0, "Padding should be zero"
    assert padded_hw[4, 4] == 1.0, "Original data should be preserved"

    # Test 3D channels-first (C, H, W)
    img_chw = np.ones((3, 8, 8))
    padded_chw = _pad_image(img_chw, padding=4)
    assert padded_chw.shape == (3, 16, 16), f"CHW pad shape wrong: {padded_chw.shape}"
    assert padded_chw[0, 0, 0] == 0.0, "Spatial padding should be zero"
    assert padded_chw[0, 4, 4] == 1.0, "Original data should be preserved"

    # Test 3D channels-last (H, W, C) — shape[0] > 4 triggers this path
    img_hwc = np.ones((32, 32, 3))
    padded_hwc = _pad_image(img_hwc, padding=4)
    assert padded_hwc.shape == (40, 40, 3), f"HWC pad shape wrong: {padded_hwc.shape}"

    # Test error on 1D input
    try:
        _pad_image(np.ones((10,)), padding=4)
        assert False, "Should raise ValueError for 1D input"
    except ValueError:
        pass

    print("_pad_image works correctly!")


def test_unit_random_crop_region():
    """Test _random_crop_region helper."""
    print("Unit Test: _random_crop_region...")

    padded_h, padded_w = 40, 40
    target_h, target_w = 32, 32
    max_top = padded_h - target_h   # 8
    max_left = padded_w - target_w  # 8

    # Run many trials and verify bounds
    for _ in range(200):
        top, left = _random_crop_region(padded_h, padded_w, target_h, target_w)
        assert 0 <= top <= max_top, f"top={top} out of range [0, {max_top}]"
        assert 0 <= left <= max_left, f"left={left} out of range [0, {max_left}]"

    # Verify both endpoints are reachable (statistical test)
    tops = set()
    lefts = set()
    for _ in range(500):
        t, l = _random_crop_region(padded_h, padded_w, target_h, target_w)
        tops.add(t)
        lefts.add(l)

    assert 0 in tops, "Position 0 should be reachable"
    assert max_top in tops, f"Position {max_top} should be reachable"
    assert 0 in lefts, "Position 0 should be reachable"
    assert max_left in lefts, f"Position {max_left} should be reachable"

    # Edge case: when padded == target, only one valid position
    top, left = _random_crop_region(32, 32, 32, 32)
    assert top == 0 and left == 0, "Only valid position when padded == target"

    print("_random_crop_region works correctly!")


def test_unit_augmentation():
    """Test data augmentation transforms."""
    print("Unit Test: Data Augmentation...")

    # Test 1: RandomHorizontalFlip
    print("  Testing RandomHorizontalFlip...")
    flip = RandomHorizontalFlip(p=1.0)  # Always flip for deterministic test

    img = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 image
    flipped = flip(img)
    expected = np.array([[3, 2, 1], [6, 5, 4]])
    assert np.array_equal(flipped, expected), f"Flip failed: {flipped} vs {expected}"

    # Test never flip
    no_flip = RandomHorizontalFlip(p=0.0)
    unchanged = no_flip(img)
    assert np.array_equal(unchanged, img), "p=0 should never flip"

    # Test 2: RandomCrop shape preservation
    print("  Testing RandomCrop...")
    crop = RandomCrop(32, padding=4)

    # Test with (C, H, W) format (CIFAR-10 style)
    img_chw = rng.standard_normal((3, 32, 32))
    cropped = crop(img_chw)
    assert cropped.shape == (3, 32, 32), f"CHW crop shape wrong: {cropped.shape}"

    # Test with (H, W) format
    img_hw = rng.standard_normal((28, 28))
    crop_hw = RandomCrop(28, padding=4)
    cropped_hw = crop_hw(img_hw)
    assert cropped_hw.shape == (28, 28), f"HW crop shape wrong: {cropped_hw.shape}"

    # Test 3: Compose pipeline
    print("  Testing Compose...")
    transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomCrop(32, padding=4)
    ])

    img = rng.standard_normal((3, 32, 32))
    augmented = transforms(img)
    assert augmented.shape == (3, 32, 32), f"Compose output shape wrong: {augmented.shape}"

    # Test 4: Transforms work with Tensor
    print("  Testing Tensor compatibility...")
    tensor_img = Tensor(rng.standard_normal((3, 32, 32)))

    flip_result = RandomHorizontalFlip(p=1.0)(tensor_img)
    assert isinstance(flip_result, Tensor), "Flip should return Tensor when given Tensor"

    crop_result = RandomCrop(32, padding=4)(tensor_img)
    assert isinstance(crop_result, Tensor), "Crop should return Tensor when given Tensor"

    # Test 5: Randomness verification
    print("  Testing randomness...")
    flip_random = RandomHorizontalFlip(p=0.5)

    # Run many times and check we get both outcomes
    flips = 0
    no_flips = 0
    test_img = np.array([[1, 2]])

    for _ in range(100):
        result = flip_random(test_img)
        if np.array_equal(result, np.array([[2, 1]])):
            flips += 1
        else:
            no_flips += 1

    # With p=0.5, we should get roughly 50/50 (allow for randomness)
    assert flips > 20 and no_flips > 20, f"Flip randomness seems broken: {flips} flips, {no_flips} no-flips"

    print("Data Augmentation works correctly!")


def test_unit_dataloader():
    """Test DataLoader implementation."""
    print("Unit Test: DataLoader...")

    # Create test dataset
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # 5 samples
    labels = Tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(features, labels)

    # Test basic batching (no shuffle)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Test length calculation
    assert len(loader) == 3, f"Expected 3 batches, got {len(loader)}"  # ceil(5/2) = 3

    batches = list(loader)
    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

    # Test first batch
    batch_features, batch_labels = batches[0]
    assert batch_features.data.shape == (2, 2), f"Wrong batch features shape: {batch_features.data.shape}"
    assert batch_labels.data.shape == (2,), f"Wrong batch labels shape: {batch_labels.data.shape}"

    # Test last batch (should have 1 sample)
    batch_features, batch_labels = batches[2]
    assert batch_features.data.shape == (1, 2), f"Wrong last batch features shape: {batch_features.data.shape}"
    assert batch_labels.data.shape == (1,), f"Wrong last batch labels shape: {batch_labels.data.shape}"

    # Test that data is preserved
    assert np.array_equal(batches[0][0].data[0], [1, 2]), "First sample should be [1,2]"
    assert batches[0][1].data[0] == 0, "First label should be 0"

    # Test shuffling produces different order
    loader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
    loader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)

    batch_shuffle = list(loader_shuffle)[0]
    batch_no_shuffle = list(loader_no_shuffle)[0]

    # Note: This might occasionally fail due to random chance, but very unlikely
    # We'll just test that both contain all the original data
    shuffle_features = set(tuple(row) for row in batch_shuffle[0].data)
    no_shuffle_features = set(tuple(row) for row in batch_no_shuffle[0].data)
    expected_features = {(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)}

    assert shuffle_features == expected_features, "Shuffle should preserve all data"
    assert no_shuffle_features == expected_features, "No shuffle should preserve all data"

    print("DataLoader works correctly!")


def test_unit_dataloader_deterministic():
    """Test DataLoader deterministic shuffling with fixed seed."""
    print("Unit Test: DataLoader Deterministic Shuffling...")

    # Create test dataset
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    labels = Tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)

    # Test that same seed produces same shuffle
    random.seed(42)
    loader1 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches1 = list(loader1)

    random.seed(42)
    loader2 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches2 = list(loader2)

    # Should produce identical batches with same seed
    for i, (batch1, batch2) in enumerate(zip(batches1, batches2)):
        assert np.array_equal(batch1[0].data, batch2[0].data), \
            f"Batch {i} features should be identical with same seed"
        assert np.array_equal(batch1[1].data, batch2[1].data), \
            f"Batch {i} labels should be identical with same seed"

    # Test that different seeds produce different shuffles
    random.seed(42)
    loader3 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches3 = list(loader3)

    random.seed(123)  # Different seed
    loader4 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches4 = list(loader4)

    # Should produce different batches with different seeds (very likely)
    different = False
    for batch3, batch4 in zip(batches3, batches4):
        if not np.array_equal(batch3[0].data, batch4[0].data):
            different = True
            break

    assert different, "Different seeds should produce different shuffles"

    print("Deterministic shuffling works correctly!")


def test_unit_training_integration():
    """Test DataLoader integration with training workflow."""
    print("Integration Test: Training Workflow...")

    # Create a realistic dataset
    num_samples = 1000
    num_features = 20
    num_classes = 5

    # Synthetic classification data
    features = Tensor(rng.standard_normal((num_samples, num_features)))
    labels = Tensor(rng.integers(0, num_classes, num_samples))

    dataset = TensorDataset(features, labels)

    # Create train/val splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Manual split (in production, you'd use proper splitting utilities)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    # Create subset datasets
    train_samples = [dataset[i] for i in train_indices]
    val_samples = [dataset[i] for i in val_indices]

    # Convert back to tensors for TensorDataset
    train_features = Tensor(np.stack([sample[0].data for sample in train_samples]))
    train_labels = Tensor(np.stack([sample[1].data for sample in train_samples]))
    val_features = Tensor(np.stack([sample[0].data for sample in val_samples]))
    val_labels = Tensor(np.stack([sample[1].data for sample in val_samples]))

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Dataset splits:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Simulate training loop
    print("\n🏃 Simulated Training Loop:")

    epoch_samples = 0
    batch_count = 0

    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_count += 1
        epoch_samples += len(batch_features.data)

        # Simulate forward pass (just check shapes)
        assert batch_features.data.shape[0] <= batch_size, "Batch size exceeded"
        assert batch_features.data.shape[1] == num_features, "Wrong feature count"
        assert len(batch_labels.data) == len(batch_features.data), "Mismatched batch sizes"

        if batch_idx < 3:  # Show first few batches
            print(f"  Batch {batch_idx + 1}: {batch_features.data.shape[0]} samples")

    print(f"  Total: {batch_count} batches, {epoch_samples} samples processed")

    # Validate that all samples were seen
    assert epoch_samples == len(train_dataset), f"Expected {len(train_dataset)}, processed {epoch_samples}"

    print("Training integration works correctly!")
