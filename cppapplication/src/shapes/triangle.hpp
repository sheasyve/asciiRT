#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "../utils/ray.hpp"
#include "../utils/util.hpp"

class Triangle {
public:
    Eigen::Vector3d p1, p2, p3;
    Triangle(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3);
    std::optional<std::tuple<double, Eigen::Vector3d>> intersects(const Ray& ray) const;
    Eigen::Vector3d normal() const;
};

#endif