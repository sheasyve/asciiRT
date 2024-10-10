#ifndef INTERSECT_CUH
#define INTERSECT_CUH
#include "util.cuh"
#include "ray.cuh"
#include "../shapes/triangle.cuh"
#include "../shapes/sphere.cuh"
#include "../shapes/mesh.cuh"


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