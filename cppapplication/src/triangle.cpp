#include "triangle.hpp"

using namespace Eigen;

Triangle::Triangle(Vector3d p1, Vector3d p2, Vector3d p3) {
    this->p1 = p1;
    this->p2 = p2;
    this->p3 = p3;
}

std::optional<std::tuple<double, Vector3d>> Triangle::intersects(const Ray& ray) {
    // from wikipedia implementation of the Moller-Trumbore algorithm
    const double eps = 0.00001;
    Vector3d e12 = p2 - p1;
    Vector3d e13 = p3 - p1;
    Vector3d ray_cross_e13 = ray.direction.cross(e13);
    double det = e12.dot(ray_cross_e13);
    if (det > -eps && det < eps) return {};
    double inv_det = 1.0/det;
    Vector3d s = ray.origin - p1;
    double u = inv_det * s.dot(ray_cross_e13);
    if (u < 0 || u > 1) return {};
    Vector3d s_cross_e12 = s.cross(e12);
    double v = inv_det * ray.direction.dot(s_cross_e12);
    if (v < 0 || u + v > 1) return {};

    float t = inv_det * e13.dot(s_cross_e12);
    if (t > eps) {
        return std::make_tuple(t, ray.origin + ray.direction * t);
    } else {
        return {};
    }
}

Vector3d Triangle::normal() {
    return (this->p2 - this->p1).cross(this->p3 - this->p1).normalized();
}