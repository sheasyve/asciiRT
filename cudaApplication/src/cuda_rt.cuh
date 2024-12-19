#ifndef CUDA_RT_CUH
#define CUDA_RT_CUH

#include "utils/cuda_rt_util.cuh"

float* h_raytrace(std::vector<Mesh> meshes, int width, int height, 
    std::vector<V3f> light_positions, std::vector<V4f> light_colors, 
    float focal_length, float field_of_view, V3f camera_position);

#endif