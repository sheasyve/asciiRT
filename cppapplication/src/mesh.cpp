#include "mesh.hpp"
#include "intersectable.hpp"
#include "ray.hpp"
#include "obj.hpp"
#include "util.hpp"

Mesh::Mesh(const std::vector<std::shared_ptr<Triangle>>& tris)
    : triangles(tris) {}

// Mesh::intersects implementation
std::optional<std::tuple<double, Eigen::Vector3d>> Mesh::intersects(const Ray& ray) const {
    std::optional<std::tuple<double, Eigen::Vector3d>> nearest_intersection = std::nullopt;
    double min_t = std::numeric_limits<double>::infinity();

    // Delegate the intersection check to each Triangle in the Mesh
    for (const auto& tri : triangles) {
        auto res = tri->intersects(ray);  // Use the Triangle's intersects method
        if (res.has_value()) {
            double t = std::get<0>(res.value());  // Get the intersection distance
            if (t < min_t) {
                min_t = t;
                nearest_intersection = res;  // Update nearest intersection
            }
        }
    }

    return nearest_intersection;  // Return the closest intersection
}

std::shared_ptr<Mesh> get_mesh(const Obj& obj) {
    std::vector<std::shared_ptr<Triangle>> triangles;

    for (const auto& object : obj.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(object)) {
            triangles.push_back(tri);
        }
    }

    return std::make_shared<Mesh>(triangles);
}

