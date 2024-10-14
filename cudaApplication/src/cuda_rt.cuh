#ifndef CUDA_RT_CUH
#define CUDA_RT_CUH
#include "cuda_rt.cuh"
#include "shapes/triangle.cuh"
#include "utils/ray.cuh"
#include "shapes/mesh.cuh"

double* h_raytrace(Ray* rays, std::vector<Mesh> meshes, int width, int height, std::vector<Eigen::Vector3d> light_positions, std::vector<Eigen::Vector4d> light_colors);

#endif