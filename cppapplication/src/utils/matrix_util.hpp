#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "util.hpp"
#include "../shapes/mesh.hpp"
#include "../shapes/triangle.hpp"

std::vector<Triangle> rotate_mesh(Mesh& mesh, float rX, float rY, float rZ);
std::vector<Triangle> translate_mesh(Mesh& mesh, float tx, float ty, float tz);

#endif // MATRIX_CUH