#include "sphere.hpp"

Sphere::Sphere(const Eigen::Vector3f& center, float radius) : center(center), radius(radius) {}

std::optional<std::tuple<float, Eigen::Vector3f>> Sphere::intersects(const Ray& ray) const {
    Eigen::Vector3f oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0 * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return std::nullopt;
    } else {
        float sqrt_d = std::sqrt(discriminant);
        float t1 = (-b - sqrt_d) / (2.0 * a);
        float t2 = (-b + sqrt_d) / (2.0 * a);
        float t = (t1 > 1e-8) ? t1 : t2;
        if (t > 1e-8) {
            Eigen::Vector3f hit_point = ray.origin + t * ray.direction;
            return std::make_tuple(t, hit_point);
        }
    }
    return std::nullopt;
}