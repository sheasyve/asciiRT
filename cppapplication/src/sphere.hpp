#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "intersectable.hpp"
#include <Eigen/Dense>

class Sphere : public Intersectable {
public:
    Eigen::Vector3d center;
    double radius;

    Sphere(const Eigen::Vector3d& center, double radius);

    std::optional<std::tuple<double, Eigen::Vector3d>> intersects(const Ray& ray) const override;

    virtual ~Sphere() = default;
};

#endif
