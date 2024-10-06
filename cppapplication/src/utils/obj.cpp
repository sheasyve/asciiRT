// obj.cpp
#include "obj.hpp"

Obj::Obj(const Eigen::Matrix4d& transform) {
    std::vector<Eigen::Vector4d> points;
    points.push_back(Eigen::Vector4d(0., 0., 0., 0.));  // .obj is one-indexed
    for (std::string line; std::getline(std::cin, line);) {
        std::stringstream line_stream(line);
        std::string prefix;
        line_stream >> prefix;

        if (prefix == "v") {
            double x, y, z;
            line_stream >> x >> y >> z;
            points.push_back(transform * Eigen::Vector4d(x, y, z, 1.));

        } else if (prefix == "f") {
            // Read face data, potentially with vt/vn (format: v/vt/vn)
            std::vector<int> vertex_indices;
            // Parse each part of the face line (v/vt/vn or just v)
            std::string vertex;
            while (line_stream >> vertex) {
                std::stringstream vertex_stream(vertex);
                std::string vertex_index_str;
                // Get the vertex index (first part before any '/' character)
                std::getline(vertex_stream, vertex_index_str, '/');
                int vertex_index = std::stoi(vertex_index_str) - 1;  // Convert to 0-indexed
                vertex_indices.push_back(vertex_index);
            }
            // Ensure we have 3 vertices for a triangle (ignore quads, polygons, etc.)
            if (vertex_indices.size() == 3) {
                int ix = vertex_indices[0];
                int iy = vertex_indices[1];
                int iz = vertex_indices[2];
                if (ix >= 0 && ix < points.size() && iy >= 0 && iy < points.size() && iz >= 0 && iz < points.size()) {
                    Triangle triangle(points[ix].head<3>(), points[iy].head<3>(), points[iz].head<3>());
                    objects.push_back(triangle); // Add triangle to objects vector
                }
            }
        }
    }
}

void Obj::add_sphere(const Eigen::Vector3d& center, double radius) {
    Sphere sphere(center, radius);
    objects.push_back(sphere);  // Store sphere in the objects vector
}

Mesh Obj::get_mesh() const {//Return triangles from mesh
    std::vector<Triangle> triangles;
    for (const auto& object : objects) {
        if (std::holds_alternative<Triangle>(object)) {
            triangles.push_back(std::get<Triangle>(object));
        }
    }
    return Mesh(triangles); 
}