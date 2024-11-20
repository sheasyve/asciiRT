#ifndef V4F_CUH
#define V4F_CUH

#include <cmath>
#include <algorithm>
#include "v3f.cuh"

struct V4f {
    float x, y, z, w;

    __device__ __host__ inline V4f() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    __device__ __host__ inline V4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    // Basic operations
    __device__ __host__ inline V4f operator+(const V4f& other) const {
        return V4f(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    __device__ __host__ inline V4f operator-(const V4f& other) const {
        return V4f(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    __device__ __host__ inline V4f operator*(float scalar) const {
        return V4f(x * scalar, y * scalar, z * scalar, w * scalar);
    }

    __device__ __host__ inline V4f operator/(float scalar) const {
        float inv = 1.0f / scalar;
        return V4f(x * inv, y * inv, z * inv, w * inv);
    }

    __device__ __host__ inline V4f& operator+=(const V4f& other) {
        x += other.x; y += other.y; z += other.z; w += other.w;
        return *this;
    }

    __device__ __host__ inline V4f& operator-=(const V4f& other) {
        x -= other.x; y -= other.y; z -= other.z; w -= other.w;
        return *this;
    }

    __device__ __host__ inline float dot(const V4f& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    __device__ __host__ inline float norm() const {
        return sqrtf(dot(*this));
    }

    // Normalization
    __device__ __host__ inline V4f normalized() const {
        float n = norm();
        return (n > 0.0f) ? (*this / n) : V4f(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __device__ __host__ inline V3f to_v3f() const {
        return V3f(x, y, z);
    }
};

#endif // V4F_CUH
