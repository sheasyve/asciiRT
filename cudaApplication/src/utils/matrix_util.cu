#include "m4f.cuh"
#include "v3f.cuh"
#include "../shapes/mesh.cuh"

std::vector<Triangle> rotate_mesh(Mesh& mesh, float rX, float rY, float rZ) {
    M4f rotMatX, rotMatY, rotMatZ;
    rotMatX.m[1][1] = cosf(rX); rotMatX.m[1][2] = -sinf(rX);
    rotMatX.m[2][1] = sinf(rX); rotMatX.m[2][2] = cosf(rX);
    rotMatY.m[0][0] = cosf(rY); rotMatY.m[0][2] = sinf(rY);
    rotMatY.m[2][0] = -sinf(rY); rotMatY.m[2][2] = cosf(rY);
    rotMatZ.m[0][0] = cosf(rZ); rotMatZ.m[0][1] = -sinf(rZ);
    rotMatZ.m[1][0] = sinf(rZ); rotMatZ.m[1][1] = cosf(rZ);
    M4f rotationMatrix = rotMatZ * rotMatY * rotMatX;
    // Apply rotation to each triangle in the mesh
    std::vector<Triangle> rotated_triangles = mesh.triangles;
    for (auto& tri : rotated_triangles) {
        V4f p1 = rotationMatrix * V4f(tri.p1.x, tri.p1.y, tri.p1.z, 1.0f);
        V4f p2 = rotationMatrix * V4f(tri.p2.x, tri.p2.y, tri.p2.z, 1.0f);
        V4f p3 = rotationMatrix * V4f(tri.p3.x, tri.p3.y, tri.p3.z, 1.0f);

        tri.p1 = V3f(p1.x, p1.y, p1.z);
        tri.p2 = V3f(p2.x, p2.y, p2.z);
        tri.p3 = V3f(p3.x, p3.y, p3.z);
    }
    return rotated_triangles;
}

std::vector<Triangle> translate_mesh(Mesh& mesh, float tx, float ty, float tz) {
    // Translation vector
    V3f translation(tx, ty, tz);

    // Apply translation to each triangle in the mesh
    std::vector<Triangle> translated_triangles = mesh.triangles;
    for (auto& tri : translated_triangles) {
        tri.p1 += translation;
        tri.p2 += translation;
        tri.p3 += translation;
    }
    return translated_triangles;
}
