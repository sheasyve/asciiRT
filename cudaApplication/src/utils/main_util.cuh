#ifndef MAIN_UTIL_CUH
#define MAIN_UTIL_CUH

#include "util.cuh"
#include "load_mesh.cuh"
#include "matrix_util.cuh"
#include "ascii_print.cuh"
#include "../cuda_rt.cuh"
#include "../shapes/triangle.cuh"
#include "../shapes/mesh.cuh"

Mesh input_mesh(const std::string& filename);
void load_meshes(int argc, char* argv[], std::vector<Mesh>& meshes);
void warmup_cuda();
#endif //MAIN_UTIL_HPP