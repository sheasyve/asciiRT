#ifndef CUDA_RT_UTIL_CUH
#define CUDA_RT_UTIL_CUH

#include "util.hpp"
#include "bvh.cuh"
#include "../shapes/triangle.cuh"
#include "../shapes/mesh.cuh"

const int MAX_STACK_SIZE = 64;

__device__ bool ray_box_intersection(
    const Eigen::Vector3f& ray_origin,
    const Eigen::Vector3f& ray_direction,
    const AlignedBox3f& bbox
);

__device__ int find_closest_triangle(
    Vector3f& ray_origin,Vector3f& ray_direction, BvhTree::Node* nodes, int root_index,
    Triangle* triangles, float& min_t
);

std::vector<Triangle> get_triangles(const std::vector<Mesh>& meshes);

#endif //CUDA_RT_UTIL_CUH
