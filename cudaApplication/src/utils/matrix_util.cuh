#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "util.hpp"
#include "../shapes/mesh.cuh"
#include "../shapes/triangle.cuh"

std::vector<Triangle> rotate_mesh(Mesh& mesh, float rX, float rY, float rZ);
std::vector<Triangle> translate_mesh(Mesh& mesh, float tx, float ty, float tz);

#endif // MATRIX_CUH