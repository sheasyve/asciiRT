#include "triangle.cuh"

Triangle::Triangle(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3)
    : p1(p1), p2(p2), p3(p3) {}

__device__ __host__
double Triangle::intersects(const Eigen::Vector3d& ray_origin, const Eigen::Vector3d& ray_direction) const {
    Eigen::Vector3d e12 = p2 - p1;
    Eigen::Vector3d e13 = p3 - p1;
    Eigen::Vector3d ray_cross_e13 = ray_direction.cross(e13);
    double det = e12.dot(ray_cross_e13);
    if (det > -1e-8 && det < 1e-8) return -1.; // No intersection
    double inv_det = 1.0 / det;
    Eigen::Vector3d s = ray_origin - p1;
    double u = inv_det * s.dot(ray_cross_e13);
    if (u < 0 || u > 1) return -1.;
    Eigen::Vector3d s_cross_e12 = s.cross(e12);
    double v = inv_det * ray_direction.dot(s_cross_e12);
    if (v < 0 || u + v > 1) return -1.;
    double t = inv_det * e13.dot(s_cross_e12);
    return t;
}

__device__ __host__
Eigen::Vector3d Triangle::normal() const {
    return (p2 - p1).cross(p3 - p1).normalized();
}