// obj.hpp
#ifndef OBJ_HPP
#define OBJ_HPP

#include "triangle.hpp"
#include "sphere.hpp"
#include "mesh.hpp"
#include "util.hpp"


class Obj {
public:
    std::vector<std::variant<Triangle, Sphere, Mesh>> objects;

    // Constructor that takes a 4x4 transformation matrix
    Obj(const Eigen::Matrix4d& transform);

    // Function to add a sphere
    void add_sphere(const Eigen::Vector3d& center, double radius);

    // Function to get Mesh from Obj
    Mesh get_mesh() const;
};

#endif
