#ifndef CUDA_RT_UTIL_CUH
#define CUDA_RT_UTIL_CUH

#include "util.cuh"
#include "bvh.cuh"
#include "../shapes/triangle.cuh"
#include "../shapes/mesh.cuh"
#include "v3f.cuh"
#include "v4f.cuh"
#include "aabb.cuh"

const int MAX_STACK_SIZE = 64;

__device__ bool ray_box_intersection(
    const V3f& ray_origin,
    const V3f& ray_direction,
    const AABB& bbox
);

__device__ int find_closest_triangle(
    const V3f& ray_origin,
    const V3f& ray_direction,
    AABB* nodes_bbox,
    int* nodes_left,
    int* nodes_right,
    int* nodes_triangle,
    int root_index,
    const Triangle* triangles,
    float& min_t
);

std::vector<Triangle> get_triangles(const std::vector<Mesh>& meshes);

#endif //CUDA_RT_UTIL_CUH
