#ifndef MAIN_UTIL_CUH
#define MAIN_UTIL_CUH

#include "util.hpp"
#include "load_mesh.cuh"
#include "matrix_util.cuh"
#include "ascii_print.hpp"
#include "../cuda_rt.cuh"
#include "../shapes/triangle.cuh"
#include "../shapes/mesh.cuh"

Mesh input_mesh(const std::string& filename);
void load_meshes(int argc, char* argv[], std::vector<Mesh>& meshes);

#endif //MAIN_UTIL_HPP