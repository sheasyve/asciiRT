#ifndef MESH_HPP
#define MESH_HPP

#include "util.hpp"
#include "intersectable.hpp"
#include "triangle.hpp"

class Obj; //Forward Declaration

class Mesh : public Intersectable {
public:
    std::vector<std::shared_ptr<Triangle>> triangles;
    Mesh(const std::vector<std::shared_ptr<Triangle>>& tris);
    std::optional<std::tuple<double, Eigen::Vector3d>> intersects(const Ray& ray) const override;
};

std::shared_ptr<Mesh> get_mesh(const Obj& obj);

#endif
