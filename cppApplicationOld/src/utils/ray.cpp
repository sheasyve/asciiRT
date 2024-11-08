#include "ray.hpp"

Ray::Ray(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction)
    : origin(origin), direction(direction) {}
