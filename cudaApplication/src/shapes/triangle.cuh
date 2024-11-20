#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "../utils/util.cuh"

class Triangle {
public:
    V3f p1, p2, p3;

    __device__ __host__ Triangle(V3f p1, V3f p2, V3f p3);

    __device__ __host__ float intersects(const V3f& ray_origin, const V3f& ray_direction) const;

    __device__ __host__ V3f normal() const;
};

#endif
