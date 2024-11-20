#include "cuda_rt_util.cuh"

#include <cfloat> // For FLT_MAX

__device__ bool ray_box_intersection(const V3f& ray_origin, const V3f& ray_direction, const AABB& bbox) {
    float tmin = -FLT_MAX, tmax = FLT_MAX;
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / ray_direction[i];
        float t0 = (bbox.min[i] - ray_origin[i]) * invD;
        float t1 = (bbox.max[i] - ray_origin[i]) * invD;
        
        if (invD < 0.0f) {
            float temp = t0; t0 = t1; t1 = temp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);

        if (tmax < tmin) return false;
    }
    return true;
}

__device__ int find_closest_triangle(
    const V3f& ray_origin, const V3f& ray_direction,
    AABB* nodes_bbox, int* nodes_left, int* nodes_right, int* nodes_triangle,
    int root_index, const Triangle* triangles, float& min_t)
{
    const int MAX_STACK_SIZE = 64;
    int dfs_stack[MAX_STACK_SIZE];
    int stack_index = 0;
    int min_index = -1;

    dfs_stack[stack_index++] = root_index;
    while (stack_index > 0) {
        int node_index = dfs_stack[--stack_index];

        if (node_index < 0) continue;

        if (ray_box_intersection(ray_origin, ray_direction, nodes_bbox[node_index])) {
            if (nodes_left[node_index] == -1 && nodes_right[node_index] == -1) { // Leaf node
                int tri_idx = nodes_triangle[node_index];
                float t = triangles[tri_idx].intersects(ray_origin, ray_direction.normalized());
                if (t > 0.0f && t < min_t) { // Update closest triangle
                    min_t = t;
                    min_index = tri_idx;
                }
            } else { // Internal node, push children onto stack
                if (nodes_right[node_index] != -1 && stack_index < MAX_STACK_SIZE)
                    dfs_stack[stack_index++] = nodes_right[node_index];
                if (nodes_left[node_index] != -1 && stack_index < MAX_STACK_SIZE)
                    dfs_stack[stack_index++] = nodes_left[node_index];
            }
        }
    }
    return min_index;
}


std::vector<Triangle> get_triangles(const std::vector<Mesh>& meshes) {
    // Extracts all triangles from all input meshes
    std::vector<Triangle> triangles;
    for (const auto& mesh : meshes) {
        triangles.insert(triangles.end(), mesh.triangles.begin(), mesh.triangles.end());
    }
    return triangles;
}
