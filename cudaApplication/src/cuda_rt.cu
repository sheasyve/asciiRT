#include "cuda_rt.cuh"

__global__ void d_raytrace(
    Ray* rays, BvhTree::Node* nodes, int root_index, Triangle* triangles,
    double* output,
    int width, int height,
    Eigen::Vector3d* light_positions,
    Eigen::Vector4d* light_colors, int num_lights
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;//Ray index for this thread
    Ray r = rays[idx];
    double min_t = INF;
    int mindex = find_closest_triangle(r, nodes, root_index, triangles, min_t);//Done in cuda_rt_util.cu by traversing the BVH tree

    if (mindex == -1) {
        output[idx] = 0.0;
        return;
    }

    Eigen::Vector3d p = r.origin + r.direction * min_t;
    Triangle closest = triangles[mindex];
    Eigen::Vector3d N = closest.normal();
    N.normalize();
    Eigen::Vector3d V = -r.direction;
    V.normalize();

    double brightness = 0.005; // Start with ambient light
    double diffuse_intensity = 0.4;
    double specular_intensity = 0.4;
    double shine = 32.0;

    for (int i = 0; i < num_lights; i++) { // Add light from each source
        Eigen::Vector3d L = (light_positions[i] - p);
        double d = L.norm();
        L.normalize();
        Eigen::Vector3d light_rgb = light_colors[i].head<3>();
        double a = 1.0, b = 0.1, c = 0.01;
        double attenuation = 1.0 / (a + b * d + c * d * d); // Distance dropoff
        // Diffuse
        double lambertian = N.dot(L);
        lambertian = fmax(lambertian, 0.0);
        brightness += attenuation * diffuse_intensity * lambertian * light_rgb.norm();
        // Specular
        Eigen::Vector3d R = (2.0 * N.dot(L) * N - L).normalized();
        double spec_angle = R.dot(V);
        spec_angle = fmax(spec_angle, 0.0);
        double specular = pow(spec_angle, shine);
        brightness += attenuation * specular_intensity * specular * light_rgb.norm();
    }
    brightness = fmin(brightness, 1.0);
    output[idx] = brightness;
}

double* h_raytrace(
    Ray* rays, std::vector<Mesh> meshes,
    int width, int height,
    std::vector<Eigen::Vector3d> light_positions,
    std::vector<Eigen::Vector4d> light_colors
) {
    int size = width * height;
    int num_lights = static_cast<int>(light_positions.size());

    std::vector<Triangle> triangles = get_triangles(meshes);
    int num_triangles = static_cast<int>(triangles.size());

    BvhTree bvh(triangles);//Build BVH from triangles
    std::vector<BvhTree::Node>& nodes = bvh.nodes;//Nodes of bvh tree
    int tree_size = static_cast<int>(nodes.size());
    int root = bvh.root;

    double* h_output = new double[size];

    // Device pointers
    Ray* d_rays = nullptr;
    Triangle* d_triangles = nullptr;
    BvhTree::Node* d_nodes = nullptr;
    double* d_output = nullptr;
    Eigen::Vector3d* d_lights = nullptr;
    Eigen::Vector4d* d_light_colors = nullptr;

    cudaMalloc((void**)&d_rays, size * sizeof(Ray));
    cudaMalloc((void**)&d_triangles, num_triangles * sizeof(Triangle));
    cudaMalloc((void**)&d_nodes, tree_size * sizeof(BvhTree::Node));
    cudaMalloc((void**)&d_output, size * sizeof(double));
    cudaMalloc((void**)&d_lights, num_lights * sizeof(Eigen::Vector3d));
    cudaMalloc((void**)&d_light_colors, num_lights * sizeof(Eigen::Vector4d));

    cudaMemcpy(d_rays, rays, size * sizeof(Ray), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, nodes.data(), tree_size * sizeof(BvhTree::Node), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, light_positions.data(), num_lights * sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice);
    cudaMemcpy(d_light_colors, light_colors.data(), num_lights * sizeof(Eigen::Vector4d), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,(height + blockDim.y - 1) / blockDim.y);

    d_raytrace<<<gridDim, blockDim>>>(
        d_rays, d_nodes, root, d_triangles,
        d_output,
        width, height,
        d_lights, d_light_colors, num_lights
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("CUDA error. %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(h_output, d_output, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rays);
    cudaFree(d_triangles);
    cudaFree(d_nodes);
    cudaFree(d_output);
    cudaFree(d_lights);
    cudaFree(d_light_colors);

    return h_output;
}

