#include "sphere.hpp"
#include "ray.hpp"
#include "util.hpp"

Sphere::Sphere(const Eigen::Vector3d& center, double radius)
    : center(center), radius(radius) {}

// Sphere-ray intersection 
std::optional<std::tuple<double, Eigen::Vector3d>> Sphere::intersects(const Ray& ray) const {
    Eigen::Vector3d oc = ray.origin - center; 
    double a = ray.direction.dot(ray.direction);  
    double b = 2.0 * oc.dot(ray.direction); 
    double c = oc.dot(oc) - radius * radius;  
    double discriminant = b * b - 4 * a * c;  // Calculate discriminant for quadratic solution

    if (discriminant < 0) {
        return std::nullopt;
    } else {
        double sqrt_discriminant = std::sqrt(discriminant);
        // Compute the two potential intersection points (t0, t1)
        double t1 = (-b - sqrt_discriminant) / (2.0 * a);
        double t2 = (-b + sqrt_discriminant) / (2.0 * a);
        double t = (t1 > EPS) ? t1 : t2;
        if (t < EPS) return std::nullopt;;
        Eigen::Vector3d intersection_point = ray.origin + t * ray.direction;
        return std::make_tuple(t, intersection_point);
    }
}
