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
    Obj(const Eigen::Matrix4d& transform);
    void add_sphere(const Eigen::Vector3d& center, double radius);
    Mesh get_mesh() const;
};

#endif