#include "mesh.cuh"

Mesh::Mesh(const std::vector<Triangle>& tris) : triangles(tris) {}

std::optional<std::tuple<double, Eigen::Vector3d, const Triangle*>> Mesh::intersects(const Ray& ray) const {
    std::optional<std::tuple<double, Eigen::Vector3d, const Triangle*>> nearest_hit;
    double min_t = std::numeric_limits<double>::infinity();
    for (const auto& triangle : triangles) {
        auto hit = triangle.intersects(ray);
        if (hit.has_value()) {
            double t = std::get<0>(hit.value());
            if (t < min_t) {
                min_t = t;
                nearest_hit = std::make_tuple(t, std::get<1>(hit.value()), &triangle);  
            }
        }
    }
    return nearest_hit;
}
