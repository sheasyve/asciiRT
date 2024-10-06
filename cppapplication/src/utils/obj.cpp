// obj.cpp
#include "obj.hpp"

Obj::Obj(const Eigen::Matrix4d& transform) {
    std::vector<Eigen::Vector4d> points;
    points.push_back(Eigen::Vector4d(0., 0., 0., 0.));  // .obj is one-indexed
    for (std::string line; std::getline(std::cin, line);) {// Read from stdin and populate objects (triangles)
        std::stringstream line_stream(line);
        std::string s;
        double x, y, z;
        line_stream >> s >> x >> y >> z;
        if (line[0] == 'v') {
            points.push_back(transform * Eigen::Vector4d(x, y, z, 1.));
        } else if (line[0] == 'f') {
            int ix = static_cast<int>(x) - 1;
            int iy = static_cast<int>(y) - 1;
            int iz = static_cast<int>(z) - 1;
            if (ix >= 0 && ix < points.size() && iy >= 0 && iy < points.size() && iz >= 0 && iz < points.size()) {
                Triangle triangle(points[ix].head<3>(), points[iy].head<3>(), points[iz].head<3>());
                objects.push_back(triangle); // Add triangle to objects vector
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