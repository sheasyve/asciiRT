#include "rt_util.hpp"

bool ray_box_intersection(const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction, const AlignedBox3f& bbox) {
    //Intersection of ray and axis-aligned bounding box
    float tmin = NINF, tmax = INF;
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0 / ray_direction[i];
        float t0 = (bbox.min()[i] - ray_origin[i]) * invD;
        float t1 = (bbox.max()[i] - ray_origin[i]) * invD;
        if (invD < 0.0) {// Swap t0 and t1
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }
        tmin = fmax(tmin, t0);
        tmax = fmin(tmax, t1);
        if (tmax <= tmin) {
            return false;
        }
    }
    return true;
}

int find_closest_triangle(Eigen::Vector3f& ray_origin, Eigen::Vector3f& ray_direction, 
    std::vector<BvhTree::Node>& nodes, int root_index, std::vector<Triangle> triangles, float& min_t) 
{
    int stack_index = 0;
    int min_index = -1;
    int dfs_stack[MAX_STACK_SIZE];
    dfs_stack[stack_index++] = root_index;
    
    while (stack_index > 0) {
        int node_index = dfs_stack[--stack_index];
        BvhTree::Node node = nodes[node_index];
        
        if (ray_box_intersection(ray_origin, ray_direction, node.bbox)) {
            if (node.left == -1 && node.right == -1) { // Leaf node, check for min intersection
                int tri_idx = node.triangle;
                float t = triangles[tri_idx].intersects(ray_origin, ray_direction);
                
                if (t > 0 && t < min_t) { // Found new closest intersecting triangle
                    min_t = t;
                    min_index = tri_idx;
                }
            } else {
                if (node.left != -1) dfs_stack[stack_index++] = node.left;
                if (node.right != -1) dfs_stack[stack_index++] = node.right;
                if (stack_index >= MAX_STACK_SIZE) break; // Prevent stack overflow
            }
        }
    }
    return min_index;
}


std::vector<Triangle> get_triangles(const std::vector<Mesh>& meshes) {
    //Extracts all triangles from all input meshes
    std::vector<Triangle> triangles;
    for (const auto& mesh : meshes) {
        triangles.insert(triangles.end(), mesh.triangles.begin(), mesh.triangles.end());
    }
    return triangles;
}