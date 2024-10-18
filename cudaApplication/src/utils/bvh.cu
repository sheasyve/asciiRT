#include "bvh.cuh"

AlignedBox3d bbox_from_triangle(const Eigen::Vector3d &a, const Eigen::Vector3d &b, const Eigen::Vector3d &c){
    AlignedBox3d box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}

BvhTree::BvhTree(const std::vector<Triangle>& triangles){
    // Compute centroids of all triangles
    std::vector<Eigen::Vector3d> centroids(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) centroids[i] = (triangles[i].p1 + triangles[i].p2 + triangles[i].p3) / 3.0;
    std::vector<int> indexes(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) indexes[i] = static_cast<int>(i);
    indexes = sort_triangles(centroids);
    root = build_tree(indexes, triangles);
}

void BvhTree::get_longest_axis(const std::vector<Eigen::Vector3d>& centroids){
    double xmin = std::numeric_limits<double>::infinity(), xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity(), ymax = -std::numeric_limits<double>::infinity();
    double zmin = std::numeric_limits<double>::infinity(), zmax = -std::numeric_limits<double>::infinity();

    for (const auto& c : centroids) {
        if (c.x() < xmin) xmin = c.x();
        if (c.x() > xmax) xmax = c.x();
        if (c.y() < ymin) ymin = c.y();
        if (c.y() > ymax) ymax = c.y();
        if (c.z() < zmin) zmin = c.z();
        if (c.z() > zmax) zmax = c.z();
    }

    double xd = xmax - xmin;
    double yd = ymax - ymin;
    double zd = zmax - zmin;

    if (xd >= yd && xd >= zd) longest_axis = 0;
    else if (yd >= xd && yd >= zd) longest_axis = 1;
    else longest_axis = 2;
}

// Method to sort triangles based on the longest axis
std::vector<int> BvhTree::sort_triangles(const std::vector<Eigen::Vector3d>& centroids){
    std::vector<triangle_centroid> triangles(centroids.size());
    for (size_t i = 0; i < centroids.size(); i++) {
        triangles[i].centroid = centroids[i];
        triangles[i].index = static_cast<int>(i);
    }
    get_longest_axis(centroids);
    auto compare = [this](const triangle_centroid& t1, const triangle_centroid& t2) {return t1.centroid[longest_axis] < t2.centroid[longest_axis];};
    std::sort(triangles.begin(), triangles.end(), compare);
    std::vector<int> sorted_indexes(centroids.size());
    for (size_t i = 0; i < centroids.size(); ++i) sorted_indexes[i] = triangles[i].index;
    return sorted_indexes;
}

int BvhTree::build_tree(const std::vector<int>& indexes, const std::vector<Triangle>& triangles){
    if (indexes.size() == 1) {
        Node n;
        n.triangle = indexes[0];
        const Triangle& tri = triangles[indexes[0]];
        n.bbox = bbox_from_triangle(tri.p1, tri.p2, tri.p3);
        n.left = -1;
        n.right = -1;
        n.parent = -1;
        nodes.push_back(n);
        return static_cast<int>(nodes.size()) - 1;
    }
    size_t mid = indexes.size() / 2;
    std::vector<int> left_indexes(indexes.begin(), indexes.begin() + mid);
    std::vector<int> right_indexes(indexes.begin() + mid, indexes.end());
    //Build recursively
    int left_root = build_tree(left_indexes, triangles);
    int right_root = build_tree(right_indexes, triangles);
    //Create parent node
    Node n;
    n.bbox = nodes[left_root].bbox;
    n.bbox.extend(nodes[right_root].bbox);
    n.left = left_root;
    n.right = right_root;
    n.triangle = -1;
    n.parent = -1;
    nodes.push_back(n);
    int parent_index = static_cast<int>(nodes.size()) - 1;
    nodes[left_root].parent = parent_index;
    nodes[right_root].parent = parent_index;
    return parent_index;
}
