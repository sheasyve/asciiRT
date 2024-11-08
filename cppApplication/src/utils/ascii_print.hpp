#ifndef ASCII_PRINT_HPP
#define ASCII_PRINT_HPP

#include "util.hpp"

std::pair<int, int> find_boundary(float* color, int w, int h);
void print_scene_in_ascii(float* color, int w, int h);

#endif // ASCII_PRINT_HPP