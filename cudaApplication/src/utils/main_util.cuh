#ifndef PRINT_UTIL_CUH
#define PRINT_UTIL_CUH

#include "util.cuh"
#include "ray.cuh"
#include "load_mesh.cuh"
#include "intersect.cuh"
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

#endif //PRINT_UTIL_HPP