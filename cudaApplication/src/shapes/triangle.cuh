#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "../utils/util.hpp"

class Triangle {
public:
    Eigen::Vector3d p1, p2, p3;
    Triangle(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3);
    __device__ __host__ double intersects(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_direction) const;
    __device__ __host__ Eigen::Vector3d normal() const;
};

#endif