#ifndef BVH_CUH
#define BVH_CUH

#include "util.hpp"
#include "../shapes/triangle.cuh"
#include <vector>

class BvhTree
{
public:
    // Structure of Arrays for Node properties
    class Node
    {
    public:
        AlignedBox3f* bbox;   // Array of bounding boxes
        int* parent;          // Array of parent indices
        int* left;            // Array of left child indices
        int* right;           // Array of right child indices
        int* triangle;        // Array of triangle indices
        int num_nodes;        // Total number of nodes
        Node() : bbox(nullptr), parent(nullptr), left(nullptr), right(nullptr), triangle(nullptr), num_nodes(0) {}

    void allocate(size_t size) {
        bbox = new AlignedBox3f[size];
        parent = new int[size];
        left = new int[size];
        right = new int[size];
        triangle = new int[size];
        num_nodes = 0;
    }

    void free() {
        delete[] bbox;
        delete[] parent;
        delete[] left;
        delete[] right;
        delete[] triangle;
    }
    };

    struct triangle_centroid {
        Eigen::Vector3f centroid;
        int index = 0;
    };
    Node nodes;  // SoA for all nodes
    int root;    // Root node index

    BvhTree() : root(-1) {}
    BvhTree(const std::vector<Triangle>& triangles);

    int build_tree(const std::vector<int>& indexes, const std::vector<Triangle>& triangles);

    std::vector<int> sort_triangles(const std::vector<Eigen::Vector3f>& centroids);
    void get_longest_axis(const std::vector<Eigen::Vector3f>& centroids);
    int longest_axis = 0;
};

#endif // BVH_CUH

//Tree inspired by work in CSC 305 with Teseo Schneider