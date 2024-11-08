#ifndef MESH_HPP
#define MESH_HPP

#include "../utils/util.hpp"
#include "triangle.hpp"

class Mesh {
public:
    std::vector<Triangle> triangles;
    Mesh(const std::vector<Triangle>& tris);
};

#endif // MESH_HPP