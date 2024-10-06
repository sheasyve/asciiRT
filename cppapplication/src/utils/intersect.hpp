#ifndef INTERSECT_HPP
#define INTERSECT_HPP
#include "util.hpp"
#include "ray.hpp"
#include "../shapes/triangle.hpp"
#include "../shapes/sphere.hpp"
#include "../shapes/mesh.hpp"


class Intersect {
public:
    Intersect(const Ray& ray);
    std::optional<std::tuple<double, Eigen::Vector3d>> operator()(const Triangle& triangle) const;
    std::optional<std::tuple<double, Eigen::Vector3d>> operator()(const Sphere& sphere) const;
    std::optional<std::tuple<double, Eigen::Vector3d>> operator()(const Mesh& mesh) const;

private:
    const Ray& ray;
};

#endif // INTERSECT_HPP