#include "intersect.hpp"

//This class returns the intersection based on shape type

Intersect::Intersect(const Ray& ray) : ray(ray) {}

std::optional<std::tuple<float, Eigen::Vector3f>> Intersect::operator()(const Triangle& triangle) const {
    return triangle.intersects(ray);
}

std::optional<std::tuple<float, Eigen::Vector3f>> Intersect::operator()(const Sphere& sphere) const {
    return sphere.intersects(ray);
}

std::optional<std::tuple<float, Eigen::Vector3f>> Intersect::operator()(const Mesh& mesh) const {
    std::optional<std::tuple<float, Eigen::Vector3f>> nearest_hit;
    float min_t = std::numeric_limits<float>::infinity();
    for (const auto& triangle : mesh.triangles) {
        auto hit = triangle.intersects(ray);
        if (hit.has_value() && std::get<0>(hit.value()) < min_t) {
            min_t = std::get<0>(hit.value());
            nearest_hit = hit;
        }
    }
    return nearest_hit;
}