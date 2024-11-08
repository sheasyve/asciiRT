#include "cuda_rt.cuh"

__global__ void d_raytrace(
    Eigen::Vector3f* ray_origins, Eigen::Vector3f* ray_directions, 
    BvhTree::Node* nodes, int root_index, Triangle* triangles,
    float* output,
    int width, int height,
    Eigen::Vector3f* light_positions,
    Eigen::Vector4f* light_colors, int num_lights
) {
    //Ray init
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    Eigen::Vector3f origin = ray_origins[idx];
    Eigen::Vector3f direction = ray_directions[idx];
    // Phong shading parameters
    float diffuse_intensity = 0.4;
    float specular_intensity = 0.4;
    float reflection_coefficient = 0.5; 
    float shine = 32.0;
    float a = 1.0, b = 0.1, c = 0.01;
    float brightness = 0.0;
    int max_depth = 3; //Max reflections
    for (int depth = 0; depth < max_depth; depth++) {//Perform RT up to depth times for each reflection
        float local_brightness = 0.005, min_t = INF;
        int mindex = find_closest_triangle(origin, direction, nodes, root_index, triangles, min_t);
        if (mindex == -1) {brightness += 0.0;break;}
        // Compute intersection point and normal
        Eigen::Vector3f p = origin + direction * min_t;
        Triangle closest = triangles[mindex];
        Eigen::Vector3f N = closest.normal();
        N.normalize();
        Eigen::Vector3f V = -direction;
        V.normalize();
        for (int i = 0; i < num_lights; i++) {//For each light
            Eigen::Vector3f L = light_positions[i] - p;
            float d = L.norm();
            L.normalize();
            //Shadow ray cast
            Eigen::Vector3f shadow_ray_origin = p + N * 1e-4;
            float shadow_ray_t = d; // Maximum distance to check (distance to light)
            int shadow_mindex = find_closest_triangle(shadow_ray_origin, L, nodes, root_index, triangles, shadow_ray_t);
            bool in_shadow = (shadow_mindex != -1 && shadow_ray_t > 0.0);
            float attenuation = in_shadow ? 0.0 : (1.0 / (a + b * d + c * d * d));
            Eigen::Vector3f light_rgb = light_colors[i].head<3>();
            // Diffuse
            float lambertian = fmax(N.dot(L), 0.0);
            local_brightness += attenuation * diffuse_intensity * lambertian * light_rgb.norm();
            // Specular
            Eigen::Vector3f R = (2.0 * N.dot(L) * N - L).normalized();
            float spec_angle = fmax(R.dot(V), 0.0);
            float specular = pow(spec_angle, shine);
            local_brightness += attenuation * specular_intensity * specular * light_rgb.norm();
        }
        local_brightness = fmin(local_brightness, 1.0);
        brightness += pow(reflection_coefficient, depth) * local_brightness;
        direction = direction - 2.0 * direction.dot(N) * N;//Calculate new ray direction from reflection
        direction.normalize();
        origin = p + direction * 1e-4;
    }
    output[idx] = brightness;
}

float* h_raytrace(
    std::vector<Eigen::Vector3f> ray_origins,
    std::vector<Eigen::Vector3f> ray_directions, 
    std::vector<Mesh> meshes,
    int width, int height,
    std::vector<Eigen::Vector3f> light_positions,
    std::vector<Eigen::Vector4f> light_colors
) {
    int size = width * height;
    int num_lights = static_cast<int>(light_positions.size());
    std::vector<Triangle> triangles = get_triangles(meshes);
    int num_triangles = static_cast<int>(triangles.size());
    // Build BVH
    BvhTree bvh(triangles);
    std::vector<BvhTree::Node>& nodes = bvh.nodes;
    int tree_size = static_cast<int>(nodes.size());
    int root = bvh.root;
    float* h_output = new float[size];
    //Device pointers
    Eigen::Vector3f* d_ray_origins = nullptr;
    Eigen::Vector3f* d_ray_directions = nullptr;
    Triangle* d_triangles = nullptr;
    BvhTree::Node* d_nodes = nullptr;
    float* d_output = nullptr;
    Eigen::Vector3f* d_lights = nullptr;
    Eigen::Vector4f* d_light_colors = nullptr;
    //Allocate
    cudaMalloc((void**)&d_ray_origins, size * sizeof(Eigen::Vector3f));
    cudaMalloc((void**)&d_ray_directions, size * sizeof(Eigen::Vector3f));
    cudaMalloc((void**)&d_triangles, num_triangles * sizeof(Triangle));
    cudaMalloc((void**)&d_nodes, tree_size * sizeof(BvhTree::Node));
    cudaMalloc((void**)&d_output, size * sizeof(float));
    cudaMalloc((void**)&d_lights, num_lights * sizeof(Eigen::Vector3f));
    cudaMalloc((void**)&d_light_colors, num_lights * sizeof(Eigen::Vector4f));
    //Copy
    cudaMemcpy(d_ray_origins, ray_origins.data(), size * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_directions, ray_directions.data(), size * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, nodes.data(), tree_size * sizeof(BvhTree::Node), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, light_positions.data(), num_lights * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_light_colors, light_colors.data(), num_lights * sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    //Kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    d_raytrace<<<gridDim, blockDim>>>(
        d_ray_origins, d_ray_directions, d_nodes, root, d_triangles,
        d_output,
        width, height,
        d_lights, d_light_colors, num_lights
    );
    //Copy back and free
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("CUDA error. %s\n", cudaGetErrorString(err));
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ray_origins);
    cudaFree(d_ray_directions);
    cudaFree(d_triangles);
    cudaFree(d_nodes);
    cudaFree(d_output);
    cudaFree(d_lights);
    cudaFree(d_light_colors);
    return h_output;
}

