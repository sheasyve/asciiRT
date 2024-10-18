// obj.cpp
#include "load_mesh.cuh"

LoadMesh::LoadMesh(const Eigen::Matrix4d& transform, std::istream& input_stream) {
    std::vector<Eigen::Vector4d> points;
    
    for (std::string line; std::getline(input_stream, line);) {
        std::stringstream line_stream(line);
        std::string prefix;
        line_stream >> prefix;

        if (prefix == "v") {//Read vertice
            double x, y, z;
            line_stream >> x >> y >> z;
            points.push_back(transform * Eigen::Vector4d(x, y, z, 1.0));

        } else if (prefix == "f") {//Read face
            std::vector<int> vertex_indices;
            std::string vertex;
            while (line_stream >> vertex) {
                std::stringstream vertex_stream(vertex);
                std::string vertex_index_str;
                std::getline(vertex_stream, vertex_index_str, '/');
                int vertex_index = std::stoi(vertex_index_str) - 1;  // Convert to 0-indexed
                vertex_indices.push_back(vertex_index);
            }
            // Handle triangles (3 vertices)
            if (vertex_indices.size() == 3) {
                int ix = vertex_indices[0], iy = vertex_indices[1], iz = vertex_indices[2];
                if (ix >= 0 && ix < points.size() && iy >= 0 && iy < points.size() && iz >= 0 && iz < points.size()) {
                    Triangle triangle(points[ix].head<3>(), points[iy].head<3>(), points[iz].head<3>());
                    objects.push_back(triangle);
                }
            // Handle quads (4 vertices) by splitting into two triangles
            } else if (vertex_indices.size() == 4) {
                int ix = vertex_indices[0], iy = vertex_indices[1], iz = vertex_indices[2], iw = vertex_indices[3];
                if (ix >= 0 && ix < points.size() && iy >= 0 && iy < points.size() && iz >= 0 && iz < points.size() && iw >= 0 && iw < points.size()) {
                    Triangle triangle1(points[ix].head<3>(), points[iy].head<3>(), points[iz].head<3>());
                    Triangle triangle2(points[ix].head<3>(), points[iz].head<3>(), points[iw].head<3>());
                    objects.push_back(triangle1);
                    objects.push_back(triangle2);
                }
            }
        }
    }
}

Mesh LoadMesh::get_mesh() const {
    std::vector<Triangle> triangles;
    for (const auto& object : objects) {
        if (std::holds_alternative<Triangle>(object)) {
            triangles.push_back(std::get<Triangle>(object));
        }
    }
    return Mesh(triangles); 
}

