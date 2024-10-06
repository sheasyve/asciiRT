#include "sphere.hpp"
#include "util.hpp"

Sphere::Sphere(const Eigen::Vector3d& center, double radius) : center(center), radius(radius) {}

std::optional<std::tuple<double, Eigen::Vector3d>> Sphere::intersects(const Ray& ray) const {
    Eigen::Vector3d oc = ray.origin - center;
    double a = ray.direction.dot(ray.direction);
    double b = 2.0 * oc.dot(ray.direction);
    double c = oc.dot(oc) - radius * radius;
    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return std::nullopt;
    } else {
        double sqrt_d = std::sqrt(discriminant);
        double t1 = (-b - sqrt_d) / (2.0 * a);
        double t2 = (-b + sqrt_d) / (2.0 * a);
        double t = (t1 > 1e-8) ? t1 : t2;
        if (t > 1e-8) {
            Eigen::Vector3d hit_point = ray.origin + t * ray.direction;
            return std::make_tuple(t, hit_point);
        }
    }
    return std::nullopt;
}