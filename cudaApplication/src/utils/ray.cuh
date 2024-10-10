#ifndef RAY_CUH
#define RAY_CUH

#include "util.cuh"

class Ray {
public:
    Eigen::Vector3d origin, direction;
    Ray(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction);
};

#endif  // RAY_HPP