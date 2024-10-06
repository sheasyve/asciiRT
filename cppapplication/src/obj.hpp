#ifndef OBJ_HPP
#define OBJ_HPP

#include "intersectable.hpp"
#include "util.hpp"

class Obj {
public:
    std::vector<std::shared_ptr<Intersectable>> objects;
    Obj(Matrix4d transform);
    void add_sphere(const Eigen::Vector3d& center, double radius);
};

#endif