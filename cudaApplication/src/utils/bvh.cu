#include "bvh.cuh"

AABB bbox_from_triangle(const V3f &a, const V3f &b, const V3f &c){
    AABB box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}

BvhTree::BvhTree(const std::vector<Triangle>& triangles){
    size_t max_nodes = triangles.size() * 2 - 1; // Maximum possible nodes in a binary tree
    nodes.allocate(max_nodes);
    
    // Compute centroids of all triangles once
    std::vector<V3f> centroids(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i)
        centroids[i] = (triangles[i].p1 + triangles[i].p2 + triangles[i].p3) / 3.0f;

    // Initialize indexes
    std::vector<int> indexes(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i)
        indexes[i] = static_cast<int>(i);

    // Build the tree
    root = build_tree(indexes, triangles, centroids);
}


void BvhTree::get_longest_axis(const std::vector<int>& indexes, const std::vector<V3f>& centroids) {
    float xmin = FLT_MAX, xmax = -FLT_MAX;
    float ymin = FLT_MAX, ymax = -FLT_MAX;
    float zmin = FLT_MAX, zmax = -FLT_MAX;

    for (int idx : indexes) {
        const V3f& c = centroids[idx];
        if (c.x < xmin) xmin = c.x;
        if (c.x > xmax) xmax = c.x;
        if (c.y < ymin) ymin = c.y;
        if (c.y > ymax) ymax = c.y;
        if (c.z < zmin) zmin = c.z;
        if (c.z > zmax) zmax = c.z;
    }

    float xd = xmax - xmin;
    float yd = ymax - ymin;
    float zd = zmax - zmin;

    if (xd >= yd && xd >= zd)
        longest_axis = 0;
    else if (yd >= xd && yd >= zd)
        longest_axis = 1;
    else
        longest_axis = 2;
}


std::vector<int> BvhTree::sort_triangles(const std::vector<int>& indexes, const std::vector<V3f>& centroids){
    std::vector<triangle_centroid> triangles(indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        int idx = indexes[i];
        triangles[i].centroid = centroids[idx];
        triangles[i].index = idx;
    }

    auto compare = [this](const triangle_centroid& t1, const triangle_centroid& t2) {
        return t1.centroid[longest_axis] < t2.centroid[longest_axis];
    };

    std::sort(triangles.begin(), triangles.end(), compare);

    std::vector<int> sorted_indexes(indexes.size());
    for (size_t i = 0; i < indexes.size(); ++i)
        sorted_indexes[i] = triangles[i].index;

    return sorted_indexes;
}


int BvhTree::build_tree(const std::vector<int>& indexes, const std::vector<Triangle>& triangles, const std::vector<V3f>& centroids){
    if (indexes.size() == 1) {
        int node_index = nodes.num_nodes++;
        int tri_idx = indexes[0];
        nodes.triangle[node_index] = tri_idx;
        const Triangle& tri = triangles[tri_idx];
        nodes.bbox[node_index] = bbox_from_triangle(tri.p1, tri.p2, tri.p3);
        nodes.left[node_index] = -1;
        nodes.right[node_index] = -1;
        nodes.parent[node_index] = -1;
        return node_index;
    }

    // Determine the longest axis for the current subset
    get_longest_axis(indexes, centroids);

    // Sort the current subset of indexes based on the centroids along the longest axis
    std::vector<int> sorted_indexes = sort_triangles(indexes, centroids);

    // Split the sorted indexes
    size_t mid = sorted_indexes.size() / 2;
    std::vector<int> left_indexes(sorted_indexes.begin(), sorted_indexes.begin() + mid);
    std::vector<int> right_indexes(sorted_indexes.begin() + mid, sorted_indexes.end());

    // Build the left and right subtrees recursively
    int left_root = build_tree(left_indexes, triangles, centroids);
    int right_root = build_tree(right_indexes, triangles, centroids);

    // Create parent node
    int node_index = nodes.num_nodes++;
    nodes.bbox[node_index] = nodes.bbox[left_root];
    nodes.bbox[node_index].merge(nodes.bbox[right_root]);
    nodes.left[node_index] = left_root;
    nodes.right[node_index] = right_root;
    nodes.triangle[node_index] = -1; // Internal node
    nodes.parent[node_index] = -1;
    nodes.parent[left_root] = node_index;
    nodes.parent[right_root] = node_index;

    return node_index;
}

