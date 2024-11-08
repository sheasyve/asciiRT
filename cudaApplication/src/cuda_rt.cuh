#ifndef CUDA_RT_CUH
#define CUDA_RT_CUH

#include "utils/cuda_rt_util.cuh"

float* h_raytrace(std::vector<Eigen::Vector3f> ray_origins, std::vector<Eigen::Vector3f> ray_directions, std::vector<Mesh> meshes, int width, int height, std::vector<Eigen::Vector3f> light_positions, std::vector<Eigen::Vector4f> light_colors);

#endif