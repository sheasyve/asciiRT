#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "util.hpp"
#include "../shapes/mesh.cuh"
#include "../shapes/triangle.cuh"

std::vector<Triangle> rotate_mesh(Mesh& mesh, double rX, double rY, double rZ);
std::vector<Triangle> translate_mesh(Mesh& mesh, double tx, double ty, double tz);

#endif // MATRIX_CUH