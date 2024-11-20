#include "triangle.cuh"

__device__ __host__
Triangle::Triangle(V3f p1, V3f p2, V3f p3)
    : p1(p1), p2(p2), p3(p3) {}

__device__ __host__
float Triangle::intersects(const V3f& ray_origin, const V3f& ray_direction) const {
    const float EPSILON = 1e-8f;
    // Compute edges
    V3f e12 = p2 - p1;
    V3f e13 = p3 - p1;
    // Compute determinant
    V3f ray_cross_e13 = ray_direction.cross(e13);
    float det = e12.dot(ray_cross_e13);
    if (det > -EPSILON && det < EPSILON) {
        return -1.0f; // No intersection
    }
    float inv_det = 1.0f / det;
    V3f s = ray_origin - p1;
    float u = inv_det * s.dot(ray_cross_e13);
    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }
    V3f s_cross_e12 = s.cross(e12);
    float v = inv_det * ray_direction.dot(s_cross_e12);
    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }
    float t = inv_det * e13.dot(s_cross_e12);
    if (t < EPSILON) {
        return -1.0f;
    }
    return t;
}

__device__ __host__
V3f Triangle::normal() const {
    return (p2 - p1).cross(p3 - p1).normalized();
}
