#ifndef CUDA_RT_CUH
#define CUDA_RT_CUH

#include "utils/cuda_rt_util.cuh"

double* h_raytrace(Ray* rays, std::vector<Mesh> meshes, int width, int height, std::vector<Eigen::Vector3d> light_positions, std::vector<Eigen::Vector4d> light_colors);

#endif