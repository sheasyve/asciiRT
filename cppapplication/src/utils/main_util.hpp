#ifndef PRINT_UTIL_HPP
#define PRINT_UTIL_HPP

#include "util.hpp"
#include "ray.hpp"
#include "load_mesh.hpp"
#include "intersect.hpp"
#include "matrix.hpp"
#include "../shapes/triangle.hpp"
#include "../shapes/sphere.hpp"
#include "../shapes/mesh.hpp"

std::pair<int, int> find_boundary(const Eigen::MatrixXd &Color, int w, int h) {
    int first_line = -1,last_line = -1;
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            if (Color(i, j) > 0.0) {
                if (first_line == -1) {
                    first_line = j; 
                }
                last_line = j;  
                break;
            }
        }
    }
    return {first_line, last_line};  // Return a pair of the first and last non-empty line indices
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