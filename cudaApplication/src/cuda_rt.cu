#include "cuda_rt.cuh"
#include "shapes/triangle.cuh"
#include "utils/ray.cuh"
#include "shapes/mesh.cuh"

__global__ void d_raytrace(Ray* rays, Triangle* triangles, int num_triangles, double* output, Eigen::Vector3d* light_positions, int num_lights) {
    int idx = threadIdx.x * blockDim.x + blockIdx.x;
    Ray r = rays[idx];

    double min_t = 1000000.0;
    int min_index = -1;
    for (int i = 0; i < num_triangles; i++) {
        double t = triangles[i].intersects(r);
        if (t > 0 && t < min_t) {
            min_t = t;
            min_index = i;
        }
    }

    Eigen::Vector3d p = r.origin + r.direction*min_t;

    Triangle closest = triangles[min_index];
    auto N = closest.normal();
    double brightness = 0.;
    for (int i = 0; i < num_lights; i++) {
        auto L = (light_positions[i] - p).normalized();
        double lambertian = N.dot(L);
        lambertian = lambertian < 0 ? 0 : lambertian;
        brightness += lambertian;
    }

    output[idx] = brightness;

}

double* h_raytrace(Ray* rays, Mesh mesh, int width, int height, std::vector<Eigen::Vector3d> light_positions) {
    int size = width*height;
    int num_triangles = mesh.triangles.size();
    int num_lights = light_positions.size();

    double* h_output = new double[size];
    
    Ray* d_rays = nullptr;
    Triangle* d_triangles = nullptr;
    double* d_output = nullptr;
    Eigen::Vector3d* d_lights = nullptr;
    
    cudaMalloc((void**)&d_rays, size*sizeof(Ray));
    cudaMalloc((void**)&d_triangles, num_triangles*sizeof(Triangle));
    cudaMalloc((void**)&d_output, size*sizeof(double));
    cudaMalloc((void**)&d_lights, num_lights*sizeof(Eigen::Vector3d));

    cudaMemcpy(d_rays, rays, size*sizeof(Ray), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, &mesh.triangles[0], num_triangles*sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, &light_positions[0], num_lights*sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice);

    d_raytrace<<<width, height>>>(d_rays, d_triangles, num_triangles, d_output, d_lights, num_lights);

    cudaMemcpy(h_output, d_output, size*sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_rays);
    cudaFree(d_triangles);
    cudaFree(d_output);
    
    return h_output;
}