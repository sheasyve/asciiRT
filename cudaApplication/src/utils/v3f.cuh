#ifndef V3F_CUH
#define V3F_CUH

#include <cmath>
#include <algorithm>

struct V3f {
    float x, y, z;

    __device__ __host__ V3f() : x(0.0f), y(0.0f), z(0.0f) {}
    __device__ __host__ V3f(float x, float y, float z) : x(x), y(y), z(z) {}

    // Array-like access
    __device__ __host__ float& operator[](int index) {
        if (index == 0) return x;
        if (index == 1) return y;
        return z; // Assume index == 2
    }

    __device__ __host__ const float& operator[](int index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        return z; // Assume index == 2
    }

    // Basic operations
    __device__ __host__ V3f operator+(const V3f& other) const {
        return V3f(x + other.x, y + other.y, z + other.z);
    }

    __device__ __host__ V3f operator-(const V3f& other) const {
        return V3f(x - other.x, y - other.y, z - other.z);
    }

    __device__ __host__ V3f operator-() const {
        return V3f(-x, -y, -z);
    }

    __device__ __host__ V3f operator*(float scalar) const {
        return V3f(x * scalar, y * scalar, z * scalar);
    }

    __device__ __host__ V3f operator/(float scalar) const {
        float inv = 1.0f / scalar;
        return V3f(x * inv, y * inv, z * inv);
    }

    __device__ __host__ V3f& operator+=(const V3f& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    __device__ __host__ V3f& operator-=(const V3f& other) {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }

    // Dot product
    __device__ __host__ float dot(const V3f& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Cross product
    __device__ __host__ V3f cross(const V3f& other) const {
        return V3f(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
   // Norm and normalization
    __device__ __host__ float norm() const {
        return sqrtf(dot(*this));
    }
    __device__ __host__ V3f normalized() const {
        float n = norm();
        return (n > 0.0f) ? (*this / n) : V3f(0.0f, 0.0f, 0.0f);
    }
};
    // Right scalar multiplication (V3f * scalar)
    __device__ __host__ V3f operator*(const V3f& vec, float scalar) {
        return V3f(vec.x * scalar, vec.y * scalar, vec.z * scalar);
    }
    
    // Left scalar multiplication (scalar * V3f)
    __device__ __host__ V3f operator*(float scalar, const V3f& vec) {
        return V3f(vec.x * scalar, vec.y * scalar, vec.z * scalar);
    }
    
 
#endif // V3F_CUH
