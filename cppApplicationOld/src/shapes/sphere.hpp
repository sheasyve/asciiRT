#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "../utils/util.hpp"
#include "../utils/ray.hpp"

class Sphere {
public:
    Eigen::Vector3f center;
    float radius;
    Sphere(const Eigen::Vector3f& center, float radius);
    std::optional<std::tuple<float, Eigen::Vector3f>> intersects(const Ray& ray) const;
};

#endif