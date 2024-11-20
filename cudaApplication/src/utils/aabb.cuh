#ifndef AABB_CUH
#define AABB_CUH

#include "v3f.cuh"
#include <cfloat>

// Utility function for swapping
__device__ __host__ void swap(float& a, float& b) {
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
    
    __device__ __host__ bool intersect(const V3f& ray_origin, const V3f& ray_dir_inv) const {
        float tmin = (min.x - ray_origin.x) * ray_dir_inv.x;
        float tmax = (max.x - ray_origin.x) * ray_dir_inv.x;
        if (tmin > tmax) swap(tmin, tmax);

        float tymin = (min.y - ray_origin.y) * ray_dir_inv.y;
        float tymax = (max.y - ray_origin.y) * ray_dir_inv.y;
        if (tymin > tymax) swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax)) return false;

        tmin = fmaxf(tmin, tymin);
        tmax = fminf(tmax, tymax);

        float tzmin = (min.z - ray_origin.z) * ray_dir_inv.z;
        float tzmax = (max.z - ray_origin.z) * ray_dir_inv.z;
        if (tzmin > tzmax) swap(tzmin, tzmax);

        return !(tmin > tzmax || tzmin > tmax);
    }
};

#endif // AABB_CUH
