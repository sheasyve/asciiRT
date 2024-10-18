#ifndef CUDA_RT_UTIL_CUH
#define CUDA_RT_UTIL_CUH

#include "util.hpp"
#include "ray.hpp"
#include "bvh.cuh"
#include "../shapes/triangle.cuh"
#include "../shapes/mesh.cuh"

const int MAX_STACK_SIZE = 64;

__device__ bool ray_box_intersection(
    const Eigen::Vector3d& ray_origin,
    const Eigen::Vector3d& ray_direction,
    const AlignedBox3d& bbox
);

__device__ int find_closest_triangle(
    const Ray& r, BvhTree::Node* nodes, int root_index,
    Triangle* triangles, double& min_t
);

std::vector<Triangle> get_triangles(const std::vector<Mesh>& meshes);

#endif //CUDA_RT_UTIL_CUH
