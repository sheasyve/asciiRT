#ifndef PRINT_UTIL_CUH
#define PRINT_UTIL_CUH

#include "util.cuh"
#include "ray.cuh"
#include "load_mesh.cuh"
#include "intersect.cuh"
#include "matrix.cuh"

#include "../shapes/triangle.cuh"
#include "../shapes/sphere.cuh"
#include "../shapes/mesh.cuh"

std::pair<int, int> find_boundary(double* color, int w, int h) {
    //Find the first and last non-empty lines to print without extra whitespace
    int first_line = -1,last_line = -1;
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            if (color[j*w + i] > 0.0) {
                if (first_line == -1) {
                    first_line = j; 
                }
                last_line = j;  
                break;
            }
        }
    }
    return {first_line, last_line}; 
}

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
    for (int i = 1; i < argc; ++i) {  
        std::string filename = argv[i];
        try {
            Mesh mesh = input_mesh(filename);
            meshes.push_back(mesh);
        } catch (const std::exception& e) {
            std::cerr << "Error loading mesh from " << filename << ": " << e.what() << std::endl;
        }
    }
}

#endif //PRINT_UTIL_HPP