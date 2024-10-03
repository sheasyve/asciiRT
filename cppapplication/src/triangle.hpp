#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "ray.hpp"
#include "util.hpp"

using namespace Eigen;

class Triangle {
public:
    Vector3d p1, p2, p3;
    Triangle(Vector3d p1, Vector3d p2, Vector3d p3);
    std::optional<std::tuple<double, Vector3d>> intersects(const Ray& ray);
    Vector3d normal();
};

#endif