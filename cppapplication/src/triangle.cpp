#include "triangle.hpp"
#include "util.hpp"

Triangle::Triangle(Vector3d p1, Vector3d p2, Vector3d p3)
    : p1(p1), p2(p2), p3(p3) {}  // constructor

std::optional<std::tuple<double, Vector3d>> Triangle::intersects(const Ray& ray) const {
    // Möller-Trumbore algorithm


    Vector3d e12 = p2 - p1;
    Vector3d e13 = p3 - p1;
    Vector3d ray_cross_e13 = ray.direction.cross(e13);
    double det = e12.dot(ray_cross_e13);
    if (det > -EPS && det < EPS) return {};
    double inv_det = 1.0 / det;
    Vector3d s = ray.origin - p1;
    double u = inv_det * s.dot(ray_cross_e13);
    if (u < 0 || u > 1) return {};
    Vector3d s_cross_e12 = s.cross(e12);
    double v = inv_det * ray.direction.dot(s_cross_e12);
    if (v < 0 || u + v > 1) return {};
    double t = inv_det * e13.dot(s_cross_e12);
    return (t > EPS) ? std::optional{std::make_tuple(t, ray.origin + ray.direction * t)} : std::nullopt;
}

Vector3d Triangle::normal() const {
    Vector3d normal = (p2 - p1).cross(p3 - p1).normalized();
    std::cout << "Normal: " << normal.transpose() << std::endl; // Debug output
    return normal;
}

