#ifndef MESH_CUH
#define MESH_CUH

#include <vector>
#include <optional>
#include <tuple>
#include "triangle.cuh"

class Mesh {
public:
    std::vector<Triangle> triangles;
    Mesh(const std::vector<Triangle>& tris);
};

#endif