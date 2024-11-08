#ifndef MAIN_UTIL_HPP
#define MAIN_UTIL_HPP

#include "util.hpp"
#include "load_mesh.hpp"
#include "matrix_util.hpp"
#include "ascii_print.hpp"
#include "../shapes/triangle.hpp"
#include "../shapes/mesh.hpp"
#include "bvh.hpp"
#include "rt_util.hpp"

Mesh input_mesh(const std::string& filename);
void load_meshes(int argc, char* argv[], std::vector<Mesh>& meshes);

#endif //MAIN_UTIL_HPP