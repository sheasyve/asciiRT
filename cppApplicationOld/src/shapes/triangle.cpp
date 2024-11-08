#include "triangle.hpp"

Triangle::Triangle(Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3)
    : p1(p1), p2(p2), p3(p3) {}

std::optional<std::tuple<float, Eigen::Vector3f>> Triangle::intersects(const Ray& ray) const {
    Eigen::Vector3f e12 = p2 - p1;
    Eigen::Vector3f e13 = p3 - p1;
    Eigen::Vector3f ray_cross_e13 = ray.direction.cross(e13);
    float det = e12.dot(ray_cross_e13);
    if (det > -1e-8 && det < 1e-8) return {}; // No intersection
    float inv_det = 1.0 / det;
    Eigen::Vector3f s = ray.origin - p1;
    float u = inv_det * s.dot(ray_cross_e13);
    if (u < 0 || u > 1) return {};
    Eigen::Vector3f s_cross_e12 = s.cross(e12);
    float v = inv_det * ray.direction.dot(s_cross_e12);
    if (v < 0 || u + v > 1) return {};
    float t = inv_det * e13.dot(s_cross_e12);
    return (t > 1e-8) ? std::optional<std::tuple<float, Eigen::Vector3f>>(std::make_tuple(t, ray.origin + t * ray.direction)) : std::nullopt;
}

Eigen::Vector3f Triangle::normal() const {
    return (p2 - p1).cross(p3 - p1).normalized();
}