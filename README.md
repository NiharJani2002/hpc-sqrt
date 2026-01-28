# High-Performance Square Root Implementation

Production-grade square root implementations optimized for low-latency systems. 1.08x faster than std::sqrt using SSE intrinsics and IEEE 754 bit manipulation. Proven benchmarks, comprehensive testing. Built for high-throughput numerical computation where every microsecond counts.

## What Makes This Different

Most developers use `std::sqrt` without questioning whether it's optimal for their use case. This repository demonstrates **four different approaches**, each solving specific performance bottlenecks you face in real-time computation:

- **SSE Hardware Instructions**: 1.08x faster using Intel's `rsqrtss`
- **IEEE 754 Bit Manipulation**: Reduces Newton iterations from 5-7 down to 2
- **Division-Free Computation**: Critical when division latency kills throughput
- **Portable Optimization**: Works across x86, ARM, and RISC-V architectures

## Benchmarks (10M iterations)

```
std::sqrt:         67ms  (baseline)
Newton (naive):   107ms  (1.60x slower)  ❌
SSE Fast:          62ms  (1.08x faster)  ✅
Optimal:           63ms  (1.06x faster)  ✅
```

## Why This Matters

In systems processing millions of calculations per second, small improvements compound dramatically. A 1.08x speedup across your numerical pipeline translates to:

- **Processing more data** with the same hardware
- **Lower latency** in time-sensitive operations  
- **Reduced infrastructure costs** through better efficiency

This repository demonstrates understanding of:

- Hardware-level optimization (SSE intrinsics, pipelining)
- Numerical stability and floating-point representation
- Performance engineering with proper benchmarking
- Trade-offs between speed, accuracy, and portability

## The Innovation

**Method 1: SSE Fast (`sqrt_sse_fast`)**
```cpp
// Uses Intel's hardware rsqrtss instruction
// 1.08x faster, 1.33e-05 error
__m128 val = _mm_set_ss(x);
__m128 rsqrt = _mm_rsqrt_ss(val);
```

**Method 2: Optimal (`sqrt_optimal`)**
```cpp
// IEEE 754 bit manipulation for perfect initial guess
// Only 2 Newton iterations needed
union { double d; uint64_t i; } conv = {.d = x};
conv.i = (conv.i >> 1) + (0x3ff0000000000000ULL >> 1);
```

**Method 3: Bit Hack (`sqrt_bithack`)**
```cpp
// Portable, works on any architecture
// 1.06x faster without SSE dependencies
conv.i = (1 << 29) + (conv.i >> 1) - (1 << 22);
```

## Compilation

```bash
# x86/x64 with SSE
g++ -std=c++11 -O3 -march=native -msse -msse2 sqrt.cpp -o sqrt

# ARM (Apple Silicon)
g++ -std=c++14 -O3 sqrt_portable.cpp -o sqrt
```

## Results You Can Verify

Every claim is backed by empirical testing:
- Comprehensive accuracy analysis across 12 test cases
- Speed benchmarks with 10 million iterations
- Maximum error tracking for numerical stability
- Direct comparison against `std::sqrt` baseline

## Applications

Ideal for systems requiring:
- Sub-microsecond latency constraints
- High-throughput numerical computation
- Predictable, deterministic performance
- Hardware-aware optimization

## Technical Deep Dive

The key insight: **initial guess quality determines iteration count**. 

Naive Newton starts with `guess = x`, requiring 5-7 iterations. By manipulating IEEE 754 exponent bits, we achieve ~5% initial accuracy, reducing iterations to just 2 while maintaining numerical precision.

The SSE approach goes further: hardware `rsqrtss` computes `1/sqrt(x)` in a single instruction, then one multiplication yields `sqrt(x)`.

---

*Built for systems where performance isn't optional—it's required.*

