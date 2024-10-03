#ifndef OBJ_HPP
#define OBJ_HPP

#include "triangle.hpp"

class Obj {
public:
    std::vector<Triangle> triangles;
    Obj(Matrix4d transform);
};

#endif