
typedef float4 data_t; // Force the usage of 128-bit memory accesses which is more efficient than the normal float.

__attribute__((num_compute_units(4))) // Replicate the entire hardware pipeline 4 times, ideally increasing throughput 4x
__attribute__((num_simd_work_items(1))) // Since we are using float4 manually, we set num_simd_work_items to 1 

__kernel void saxpy_kernel(
    const float a,
    __global const data_t *restrict x, 
    __global data_t *restrict y,
    const int n_vectors)
{
    int gid = get_global_id(0);
    
    #pragma unroll 4
    for (int k = 0; k < 1; k++) { 
        
        if (gid < n_vectors) {
            data_t val_x = x[gid];
            data_t val_y = y[gid];

            // SIMD execution
            data_t result;
            result.x = a * val_x.x + val_y.x;
            result.y = a * val_x.y + val_y.y;
            result.z = a * val_x.z + val_y.z;
            result.w = a * val_x.w + val_y.w;

            y[gid] = result;
        }
    }
}
