#ifndef MAIN_UTIL_CUH
#define MAIN_UTIL_CUH

#include "util.hpp"
#include "ray.hpp"
#include "load_mesh.cuh"
#include "matrix.cuh"
#include "ascii_print.hpp"
#include "../cuda_rt.cuh"
#include "../shapes/triangle.cuh"
#include "../shapes/mesh.cuh"

Mesh input_mesh(const std::string& filename) {
    std::ifstream file_stream;
    file_stream.open(filename);
    if (!file_stream) {
        std::cerr << "Can't open file: " << filename << std::endl;
        throw std::runtime_error("File opening failed.");
    }
    LoadMesh m(Eigen::Matrix4d::Identity(), file_stream);
    return m.get_mesh();
}

void load_meshes(int argc, char* argv[], std::vector<Mesh>& meshes) {
    for (int i = 1; i < argc; ++i) meshes.push_back(input_mesh(argv[i]));//Load a single mesh with the load mesh class
}

#endif //MAIN_UTIL_HPP