#ifndef RT_UTIL_HPP
#define RT_UTIL_HPP

#include "util.hpp"
#include "bvh.hpp"
#include "../shapes/triangle.hpp"
#include "../shapes/mesh.hpp"

const int MAX_STACK_SIZE = 64;

bool ray_box_intersection(
    const Eigen::Vector3f& ray_origin,
    const Eigen::Vector3f& ray_direction,
    const AlignedBox3f& bbox
);

int find_closest_triangle(Eigen::Vector3f& ray_origin, Eigen::Vector3f& ray_direction, 
    std::vector<BvhTree::Node>& nodes, int root_index, std::vector<Triangle> triangles, float& min_t) ;


std::vector<Triangle> get_triangles(const std::vector<Mesh>& meshes);

#endif //RT_UTIL_HPP
