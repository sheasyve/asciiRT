#ifndef BVH_HPP
#define BVH_HPP

#include "util.hpp"
#include "../shapes/triangle.hpp"

class BvhTree
{
public:
    class Node
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        AlignedBox3f bbox;
        int parent;   
        int left;    
        int right;    
        int triangle;

        Node() : parent(-1), left(-1), right(-1), triangle(-1) {}
    };

    struct triangle_centroid {
        Eigen::Vector3f centroid;
        int index = 0;
    };

    std::vector<Node> nodes;
    int root;

    BvhTree() = default;
    BvhTree(const std::vector<Triangle>& triangles);
    std::vector<int> sort_triangles(const std::vector<Eigen::Vector3f>& centroids);
    void get_longest_axis(const std::vector<Eigen::Vector3f>& centroids);
    int build_tree(const std::vector<int>& indexes, const std::vector<Triangle>& triangles);
    
    int longest_axis = 0;
};

#endif // BVH_HPP

//Tree inspired by work in CSC 305 with Teseo Schneider