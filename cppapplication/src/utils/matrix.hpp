#include "util.hpp"
#include "ray.hpp"
#include "../shapes/mesh.hpp"
#include "../shapes/triangle.hpp"

std::vector<Triangle> rotate_mesh(Mesh& mesh, double rX, double rY, double rZ){
    Eigen::Matrix3d rotMatX;
    rotMatX = Eigen::AngleAxisd(rX, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d rotMatY;
    rotMatY = Eigen::AngleAxisd(rY, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d rotMatZ;
    rotMatZ = Eigen::AngleAxisd(rZ, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d rotationMatrix = rotMatZ * rotMatY * rotMatX;
    std::vector<Triangle> rotated_triangles = mesh.triangles;
    for (auto& tri : rotated_triangles) {
        tri.p1 = rotationMatrix * tri.p1;
        tri.p2 = rotationMatrix * tri.p2;
        tri.p3 = rotationMatrix * tri.p3;
    }
    return rotated_triangles;
}

std::vector<Triangle> translate_mesh(Mesh& mesh, double tx, double ty, double tz) {
    Eigen::Vector3d translation(tx, ty, tz);
    std::vector<Triangle> translated_triangles = mesh.triangles;
    for (auto& tri : translated_triangles) {
        tri.p1 += translation;
        tri.p2 += translation;
        tri.p3 += translation;
    }  
    return translated_triangles;
}
