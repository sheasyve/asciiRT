#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "../utils/ray.hpp"
#include "../utils/util.hpp"

class Triangle {
public:
    Eigen::Vector3f p1, p2, p3;
    Triangle(Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3);
    std::optional<std::tuple<float, Eigen::Vector3f>> intersects(const Ray& ray) const;
    Eigen::Vector3f normal() const;
};

#endif