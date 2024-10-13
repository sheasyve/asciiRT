// obj.hpp
#ifndef LOAD_MESH_CUH
#define LOAD_MESH_CUH

#include "../shapes/triangle.cuh"
#include "../shapes/sphere.cuh"
#include "../shapes/mesh.cuh"
#include "util.cuh"

class LoadMesh {
public:
    std::vector<std::variant<Triangle, Sphere, Mesh>> objects;
    LoadMesh (const Eigen::Matrix4d& transform);
    Mesh get_mesh() const;
};

#endif // LOAD_MESH_HPP