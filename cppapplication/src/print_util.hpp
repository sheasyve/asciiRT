#ifndef PRINT_UTIL_HPP
#define PRINT_UTIL_HPP

#include "util.hpp"

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

#endif //PRINT_UTIL_HPP