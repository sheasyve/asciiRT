#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "util.hpp"
#include "ray.hpp"

class Sphere {
public:
    Eigen::Vector3d center;
    double radius;
    Sphere(const Eigen::Vector3d& center, double radius);
    std::optional<std::tuple<double, Eigen::Vector3d>> intersects(const Ray& ray) const;
};

#endif