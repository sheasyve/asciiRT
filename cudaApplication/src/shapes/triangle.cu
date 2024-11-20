#include "triangle.cuh"

__device__ __host__
Triangle::Triangle(V3f p1, V3f p2, V3f p3)
    : p1(p1), p2(p2), p3(p3) {}

__device__ __host__
float Triangle::intersects(const V3f& ray_origin, const V3f& ray_direction) const {
    // Edge vectors
    V3f edge1 = p2 - p1;
    V3f edge2 = p3 - p1;

    // Compute determinant
    V3f h = ray_direction.cross(edge2);
    float det = edge1.dot(h);

    // Cull backfacing and parallel triangles
    if (det > -1e-8 && det < 1e-8) return -1.0f;
    float inv_det = 1.0f / det;

    // Compute u parameter and test bounds
    V3f s = ray_origin - p1;
    float u = s.dot(h) * inv_det;
    if (u < 0.0f || u > 1.0f) return -1.0f;

    // Compute v parameter and test bounds
    V3f q = s.cross(edge1);
    float v = ray_direction.dot(q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    // Compute t, the distance to the intersection point
    float t = edge2.dot(q) * inv_det;

    // Return t if positive (intersection), otherwise return -1
    return (t > 0.0f) ? t : -1.0f;
}

__device__ __host__
V3f Triangle::normal() const {
    return (p2 - p1).cross(p3 - p1).normalized();
}
