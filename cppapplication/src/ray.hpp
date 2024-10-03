#ifndef RAY_HPP
#define RAY_HPP

#include "util.hpp"

using namespace Eigen;

class Ray {
public:
    Vector3d origin, direction;
    Ray(Vector3d origin, Vector3d direction);
};

#endif