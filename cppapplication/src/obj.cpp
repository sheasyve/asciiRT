#include "obj.hpp"
#include "triangle.hpp"
#include "mesh.hpp"
#include "sphere.hpp"

Obj::Obj(Matrix4d transform) {
    std::vector<Eigen::Vector4d> points;
    points.push_back(Eigen::Vector4d(0., 0., 0., 0.)); // .obj is one-indexed

    for (std::string line; std::getline(std::cin, line);) {
        std::stringstream line_stream(line);
        std::string s;
        double x, y, z;
        line_stream >> s >> x >> y >> z;

        if (line[0] == 'v') {
            // Vertex
            points.push_back(transform * Eigen::Vector4d(x, y, z, 1.));
        } else if (line[0] == 'f') {
            // Facet (triangle)
            int ix = static_cast<int>(x) - 1;
            int iy = static_cast<int>(y) - 1;
            int iz = static_cast<int>(z) - 1;
            if (ix >= 0 && ix < points.size() && iy >= 0 && iy < points.size() && iz >= 0 && iz < points.size()) {
                auto triangle = std::make_shared<Triangle>(
                    points[ix].head<3>(),
                    points[iy].head<3>(),
                    points[iz].head<3>()
                );
                objects.push_back(triangle); // Ensure triangles are stored in objects
                std::cout << "Triangle added to mesh." << std::endl; // Debug output
            }
        }
    }
}


void Obj::add_sphere(const Eigen::Vector3d& center, double radius) {
    objects.push_back(std::make_shared<Sphere>(center, radius));
    std::cout << "Added sphere with center " << center.transpose() << " and radius " << radius << std::endl;
}
