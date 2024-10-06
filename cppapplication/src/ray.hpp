#ifndef RAY_HPP
#define RAY_HPP

#include <Eigen/Dense>

class Ray {
public:
    Eigen::Vector3d origin, direction;
    Ray(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction);
};

#endif  // RAY_HPP
