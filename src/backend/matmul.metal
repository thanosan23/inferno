#include <metal_stdlib>
using namespace metal;

struct MatMulDims {
    uint m;
    uint k;
    uint n;
};

kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MatMulDims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dims.n || gid.y >= dims.m) {
        return;
    }
    float acc = 0.0;
    for (uint p = 0; p < dims.k; p++) {
        acc += a[gid.y * dims.k + p] * b[p * dims.n + gid.x];
    }
    out[gid.y * dims.n + gid.x] = acc;
}
