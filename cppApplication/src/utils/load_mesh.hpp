// obj.hpp
#ifndef LOAD_MESH_HPP
#define LOAD_MESH_HPP

#include "../shapes/triangle.hpp"
#include "../shapes/mesh.hpp"
#include "util.hpp"

class LoadMesh {
public:
    std::vector<std::variant<Triangle, Mesh>> objects;
    LoadMesh (const Eigen::Matrix4f& transform,std::istream& input_stream);
    Mesh get_mesh() const;
};

#endif // LOAD_MESH_HPP