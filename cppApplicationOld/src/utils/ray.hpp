#ifndef RAY_HPP
#define RAY_HPP

#include "util.hpp"

class Ray {
public:
    Eigen::Vector3f origin, direction;
    Ray(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction);
};

#endif  // RAY_HPP