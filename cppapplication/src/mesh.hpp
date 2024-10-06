#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <optional>
#include <tuple>
#include "triangle.hpp"
#include "ray.hpp"

class Mesh {
public:
    std::vector<Triangle> triangles;
    Mesh(const std::vector<Triangle>& tris);
    std::optional<std::tuple<double, Eigen::Vector3d, const Triangle*>> intersects(const Ray& ray) const;
};

#endif