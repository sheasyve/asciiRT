#include "mesh.hpp"

Mesh::Mesh(const std::vector<Triangle>& tris) : triangles(tris) {}

std::optional<std::tuple<float, Eigen::Vector3f, const Triangle*>> Mesh::intersects(const Ray& ray) const {
    std::optional<std::tuple<float, Eigen::Vector3f, const Triangle*>> nearest_hit;
    float min_t = std::numeric_limits<float>::infinity();
    for (const auto& triangle : triangles) {
        auto hit = triangle.intersects(ray);
        if (hit.has_value()) {
            float t = std::get<0>(hit.value());
            if (t < min_t) {
                min_t = t;
                nearest_hit = std::make_tuple(t, std::get<1>(hit.value()), &triangle);  
            }
        }
    }
    return nearest_hit;
}
