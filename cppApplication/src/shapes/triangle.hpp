#ifndef TRIANGLE_CPP
#define TRIANGLE_CPP

#include "../utils/util.hpp"

class Triangle {
public:
    Eigen::Vector3f p1, p2, p3;
    Triangle(Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3);
    float intersects(const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction) const;
    Eigen::Vector3f normal() const;
};

#endif