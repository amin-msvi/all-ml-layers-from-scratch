from tensor import Tensor

import numpy as np
rng = np.random.default_rng(7)

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024


def analyze_memory_layout():
    """Demonstrate cache effects with row vs column access patterns."""
    print("Analyzing Memory Access Patterns...")
    print("=" * 60)

    # Create a moderately-sized matrix (large enough to show cache effects)
    size = 2000
    matrix = Tensor(rng.random((size, size)))

    import time

    print(f"\nTesting with {size}×{size} matrix ({matrix.size * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB)")
    print("-" * 60)

    # Test 1: Row-wise access (cache-friendly)
    # Memory layout: [row0][row1][row2]... stored contiguously
    print("\nTest 1: Row-wise Access (Cache-Friendly)")
    start = time.time()
    row_sums = []
    for i in range(size):
        row_sum = matrix.data[i, :].sum()  # Access entire row sequentially
        row_sums.append(row_sum)
    row_time = time.time() - start
    print(f"   Time: {row_time*1000:.1f}ms")
    print("   Access pattern: Sequential (follows memory layout)")

    # Test 2: Column-wise access (cache-unfriendly)
    # Must jump between rows, poor spatial locality
    print("\nTest 2: Column-wise Access (Cache-Unfriendly)")
    start = time.time()
    col_sums = []
    for j in range(size):
        col_sum = matrix.data[:, j].sum()  # Access entire column with large strides
        col_sums.append(col_sum)
    col_time = time.time() - start
    print(f"   Time: {col_time*1000:.1f}ms")
    print(f"   Access pattern: Strided (jumps {size * BYTES_PER_FLOAT32} bytes per element)")

    # Calculate slowdown
    slowdown = col_time / row_time
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE IMPACT:")
    print(f"   Slowdown factor: {slowdown:.2f}× ({col_time/row_time:.1f}× slower)")
    print(f"   Cache misses cause {(slowdown-1)*100:.0f}% performance loss")

    # Educational insights
    print("\n💡 KEY INSIGHTS:")
    print("   1. Memory layout matters: Row-major (C-style) storage is sequential")
    print("   2. Cache lines are ~64 bytes: Row access loads nearby elements \"for free\"")
    print("   3. Column access misses cache: Must reload from DRAM every time")
    print(f"   4. This is O(n) algorithm but {slowdown:.1f}× different wall-clock time!")

    print("\n🚀 REAL-WORLD IMPLICATIONS:")
    print("   • Image processing libraries use specific memory formats for cache efficiency")
    print("   • Matrix multiplication optimized with blocking (tile into cache-sized chunks)")
    print(f"   • Transpose is expensive ({slowdown:.1f}×) because it creates a non-contiguous view with poor cache locality")
    print("   • Hardware-optimized libraries leverage memory layout for better performance")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_memory_layout()