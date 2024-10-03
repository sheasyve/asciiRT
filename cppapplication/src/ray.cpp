#include "ray.hpp"

#include "util.hpp"

using namespace Eigen;

Ray::Ray(Vector3d origin, Vector3d direction) {
    this->origin = origin;
    this->direction = direction;
}