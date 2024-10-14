// obj.hpp
#ifndef LOAD_MESH_HPP
#define LOAD_MESH_HPP

#include "../shapes/triangle.hpp"
#include "../shapes/sphere.hpp"
#include "../shapes/mesh.hpp"
#include "util.hpp"

class LoadMesh {
public:
    std::vector<std::variant<Triangle, Sphere, Mesh>> objects;
    LoadMesh(const Eigen::Matrix4d& transform, std::istream& input_stream);
    Mesh get_mesh() const;
};

#endif // LOAD_MESH_HPP