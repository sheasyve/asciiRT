#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "../utils/ray.cuh"
#include "../utils/util.cuh"

class Triangle {
public:
    Eigen::Vector3d p1, p2, p3;
    Triangle(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3);
    __device__ __host__ double intersects(const Ray& ray) const;
    __device__ __host__ Eigen::Vector3d normal() const;
};

#endif