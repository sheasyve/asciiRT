#ifndef AABB_CUH
#define AABB_CUH

#include "v3f.cuh"
#include <cfloat>

// Utility function for swapping
__device__ __host__ inline void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

struct AABB {
    V3f min;
    V3f max;

    __device__ __host__ AABB() 
        : min(V3f(FLT_MAX, FLT_MAX, FLT_MAX)),
          max(V3f(-FLT_MAX, -FLT_MAX, -FLT_MAX)) {}

    __device__ __host__ void extend(const V3f& point) {
        min.x = fminf(min.x, point.x);
        min.y = fminf(min.y, point.y);
        min.z = fminf(min.z, point.z);
        max.x = fmaxf(max.x, point.x);
        max.y = fmaxf(max.y, point.y);
        max.z = fmaxf(max.z, point.z);
    }

    __device__ __host__ void merge(const AABB& other) {
        min.x = fminf(min.x, other.min.x);
        min.y = fminf(min.y, other.min.y);
        min.z = fminf(min.z, other.min.z);
        max.x = fmaxf(max.x, other.max.x);
        max.y = fmaxf(max.y, other.max.y);
        max.z = fmaxf(max.z, other.max.z);
    }
    


};

#endif // AABB_CUH
