__kernel void saxpy_kernel(
    const float a,
    __global const float *restrict x,
    __global float *restrict y,
    const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        y[gid] = a * x[gid] + y[gid];
    }
}
