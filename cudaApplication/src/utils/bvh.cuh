#ifndef BVH_CUH
#define BVH_CUH

#include "util.hpp"
#include "../shapes/triangle.cuh"

class BvhTree
{
public:
    class Node
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        AlignedBox3d bbox;
        int parent;   
        int left;    
        int right;    
        int triangle;

        __host__ __device__ Node() : parent(-1), left(-1), right(-1), triangle(-1) {}
    };

    struct triangle_centroid {
        Eigen::Vector3d centroid;
        int index = 0;
    };

    std::vector<Node> nodes;
    int root;

    BvhTree() = default;
    BvhTree(const std::vector<Triangle>& triangles);
    std::vector<int> sort_triangles(const std::vector<Eigen::Vector3d>& centroids);
    void get_longest_axis(const std::vector<Eigen::Vector3d>& centroids);
    int build_tree(const std::vector<int>& indexes, const std::vector<Triangle>& triangles);
    
    int longest_axis = 0;
};

#endif // BVH

//Tree inspired by work in CSC 305 with Teseo Schneider