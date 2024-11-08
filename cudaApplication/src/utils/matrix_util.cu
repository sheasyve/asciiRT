#include "matrix_util.cuh"

std::vector<Triangle> rotate_mesh(Mesh& mesh, float rX, float rY, float rZ){
    Eigen::Matrix3f rotMatX;
    rotMatX = Eigen::AngleAxisd(rX, Eigen::Vector3f::UnitX());
    Eigen::Matrix3f rotMatY;
    rotMatY = Eigen::AngleAxisd(rY, Eigen::Vector3f::UnitY());
    Eigen::Matrix3f rotMatZ;
    rotMatZ = Eigen::AngleAxisd(rZ, Eigen::Vector3f::UnitZ());
    Eigen::Matrix3f rotationMatrix = rotMatZ * rotMatY * rotMatX;
    std::vector<Triangle> rotated_triangles = mesh.triangles;
    for (auto& tri : rotated_triangles) {
        tri.p1 = rotationMatrix * tri.p1;
        tri.p2 = rotationMatrix * tri.p2;
        tri.p3 = rotationMatrix * tri.p3;
    }
    return rotated_triangles;
}

std::vector<Triangle> translate_mesh(Mesh& mesh, float tx, float ty, float tz) {
    Eigen::Vector3f translation(tx, ty, tz);
    std::vector<Triangle> translated_triangles = mesh.triangles;
    for (auto& tri : translated_triangles) {
        tri.p1 += translation;
        tri.p2 += translation;
        tri.p3 += translation;
    }  
    return translated_triangles;
}