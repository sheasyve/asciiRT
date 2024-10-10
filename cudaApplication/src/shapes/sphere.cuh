#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "../utils/util.cuh"
#include "../utils/ray.cuh"

class Sphere {
public:
    Eigen::Vector3d center;
    double radius;
    Sphere(const Eigen::Vector3d& center, double radius);
    std::optional<std::tuple<double, Eigen::Vector3d>> intersects(const Ray& ray) const;
};

#endif