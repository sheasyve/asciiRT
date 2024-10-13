#include "ray.cuh"

Ray::Ray(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction)
    : origin(origin), direction(direction) {}
