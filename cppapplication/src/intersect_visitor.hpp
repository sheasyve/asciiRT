#ifndef INTERSECT_VISITOR_HPP
#define INTERSECT_VISITOR_HPP

#include "ray.hpp"
#include "triangle.hpp"
#include "sphere.hpp"
#include "mesh.hpp"
#include <optional>
#include <tuple>
#include <Eigen/Dense>

// IntersectVisitor class definition
class IntersectVisitor {
public:
    IntersectVisitor(const Ray& ray);
    std::optional<std::tuple<double, Eigen::Vector3d>> operator()(const Triangle& triangle) const;
    std::optional<std::tuple<double, Eigen::Vector3d>> operator()(const Sphere& sphere) const;
    std::optional<std::tuple<double, Eigen::Vector3d>> operator()(const Mesh& mesh) const;

private:
    const Ray& ray;
};

#endif // INTERSECT_VISITOR_HPP