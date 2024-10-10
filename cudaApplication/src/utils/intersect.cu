#include "intersect.cuh"

//This class returns the intersection based on shape type

Intersect::Intersect(const Ray& ray) : ray(ray) {}

std::optional<std::tuple<double, Eigen::Vector3d>> Intersect::operator()(const Triangle& triangle) const {
    return triangle.intersects(ray);
}

std::optional<std::tuple<double, Eigen::Vector3d>> Intersect::operator()(const Sphere& sphere) const {
    return sphere.intersects(ray);
}

std::optional<std::tuple<double, Eigen::Vector3d>> Intersect::operator()(const Mesh& mesh) const {
    std::optional<std::tuple<double, Eigen::Vector3d>> nearest_hit;
    double min_t = std::numeric_limits<double>::infinity();
    for (const auto& triangle : mesh.triangles) {
        auto hit = triangle.intersects(ray);
        if (hit.has_value() && std::get<0>(hit.value()) < min_t) {
            min_t = std::get<0>(hit.value());
            nearest_hit = hit;
        }
    }
    return nearest_hit;
}