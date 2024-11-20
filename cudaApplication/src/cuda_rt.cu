#include "cuda_rt.cuh"

__global__ void d_raytrace(
    Eigen::Vector3f *ray_origins, Eigen::Vector3f *ray_directions,
    AlignedBox3f *nodes_bbox, int *nodes_left, int *nodes_right, int *nodes_triangle, int root_index,
    Triangle *triangles, float *output,
    int width, int height,
    Eigen::Vector3f *light_positions,
    Eigen::Vector4f *light_colors, int num_lights)
{
    // Ray init
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    Eigen::Vector3f origin = ray_origins[idx];
    Eigen::Vector3f direction = ray_directions[idx];
    extern __shared__ char shared_mem[];
    Eigen::Vector3f *shared_light_positions = (Eigen::Vector3f *)shared_mem;
    Eigen::Vector4f *shared_light_colors = (Eigen::Vector4f *)(shared_mem + num_lights * sizeof(Eigen::Vector3f));
    if (threadIdx.x < num_lights){//Load lights to shared memory
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        if (threadId < num_lights) {
            shared_light_positions[threadId] = light_positions[threadId];
            shared_light_colors[threadId] = light_colors[threadId];
        }
    }
    __syncthreads();
    float brightness = 0.0;
    for (int depth = 0; depth < 3; depth++){ // Perform RT up to depth times for each reflection
        float local_brightness = 0.005, min_t = INF;
        int mindex = find_closest_triangle(origin, direction, nodes_bbox, nodes_left, nodes_right, nodes_triangle, root_index, triangles, min_t);
        if (mindex == -1) break;
        // Compute intersection point and normal
        Eigen::Vector3f p = origin + direction * min_t;
        Eigen::Vector3f N = triangles[mindex].normal().normalized();
        Eigen::Vector3f V = -direction.normalized();
        for (int i = 0; i < num_lights; i++){ // For each light
            Eigen::Vector3f L = shared_light_positions[i] - p;
            float d = L.norm();
            L.normalize();
            Eigen::Vector3f shadow_ray_origin = p + N * 1e-4;
            float shadow_ray_t = d; // Maximum distance to check (distance to light)
            int shadow_mindex = find_closest_triangle(shadow_ray_origin, L,nodes_bbox, nodes_left, nodes_right, 
                        nodes_triangle, root_index, triangles, shadow_ray_t);// Shadow ray cast
            float attenuation = (shadow_mindex != -1 && shadow_ray_t > 0.0) ? 0.0 : (1.0 / fmaf(.01, pow(d, 2), fmaf(.1, d, 1.))); // if in shadow, no light, else attenuate
            Eigen::Vector3f light_rgb = shared_light_colors[i].head<3>();
            local_brightness += attenuation * .4 * fmax(N.dot(L), 0.0) * light_rgb.norm(); //Diffuse: attenuation * diffuse * lambertarian * rgb vals
            Eigen::Vector3f R = (2.0 * N.dot(L) * N - L).normalized();
            local_brightness += attenuation * .4 * pow(fmax(R.dot(V), 0.0), 32.) * light_rgb.norm(); //Specular: attenuation * sepecular intensity * specular * rgb vals
        }
        brightness += pow(.5, depth) * fmin(local_brightness, 1.0); //(reflection_coefficient ^depth )*brightness
        direction = direction - 2.0 * direction.dot(N) * N;         // Calculate new ray direction from reflection
        direction.normalize();
        origin = p + direction * 1e-4;
    }
    output[idx] = brightness;
} // BVH nodes

float *h_raytrace(
    std::vector<Eigen::Vector3f> ray_origins,
    std::vector<Eigen::Vector3f> ray_directions,
    std::vector<Mesh> meshes,
    int width, int height,
    std::vector<Eigen::Vector3f> light_positions,
    std::vector<Eigen::Vector4f> light_colors)
{
    int size = width * height;
    int num_lights = static_cast<int>(light_positions.size());
    std::vector<Triangle> triangles = get_triangles(meshes);
    int num_triangles = static_cast<int>(triangles.size());
    // Build BVH
    BvhTree bvh(triangles);
    int tree_size = bvh.nodes.num_nodes;
    int root = bvh.root;
    float *h_output = new float[size];
    // Device pointers
    Eigen::Vector3f *d_ray_origins = nullptr;
    Eigen::Vector3f *d_ray_directions = nullptr;
    Triangle *d_triangles = nullptr;
    float *d_output = nullptr;
    Eigen::Vector3f *d_lights = nullptr;
    Eigen::Vector4f *d_light_colors = nullptr;
    AlignedBox3f *d_bbox = nullptr;
    int *d_parent = nullptr;
    int *d_left = nullptr;
    int *d_right = nullptr;
    int *d_triangle = nullptr;
    // Allocate memory
    cudaMalloc((void **)&d_bbox, tree_size * sizeof(AlignedBox3f));
    cudaMalloc((void **)&d_parent, tree_size * sizeof(int));
    cudaMalloc((void **)&d_left, tree_size * sizeof(int));
    cudaMalloc((void **)&d_right, tree_size * sizeof(int));
    cudaMalloc((void **)&d_triangle, tree_size * sizeof(int));
    cudaMalloc((void **)&d_ray_origins, size * sizeof(Eigen::Vector3f));
    cudaMalloc((void **)&d_ray_directions, size * sizeof(Eigen::Vector3f));
    cudaMalloc((void **)&d_triangles, num_triangles * sizeof(Triangle));
    cudaMalloc((void **)&d_output, size * sizeof(float));
    cudaMalloc((void **)&d_lights, num_lights * sizeof(Eigen::Vector3f));
    cudaMalloc((void **)&d_light_colors, num_lights * sizeof(Eigen::Vector4f));
    // Copy data to device
    cudaMemcpy(d_ray_origins, ray_origins.data(), size * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_directions, ray_directions.data(), size * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, light_positions.data(), num_lights * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_light_colors, light_colors.data(), num_lights * sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bbox, bvh.nodes.bbox, tree_size * sizeof(AlignedBox3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, bvh.nodes.parent, tree_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_left, bvh.nodes.left, tree_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, bvh.nodes.right, tree_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangle, bvh.nodes.triangle, tree_size * sizeof(int), cudaMemcpyHostToDevice);
    // Kernel
    int minGridSize, blockSize = 16;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, d_raytrace, 0, 0);
    dim3 blockDim(blockSize);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    size_t smem_size = num_lights * (sizeof(Eigen::Vector3f) + sizeof(Eigen::Vector4f));
    d_raytrace<<<gridDim, blockDim, smem_size>>>(
        d_ray_origins, d_ray_directions,
        d_bbox, d_left, d_right, d_triangle, root,
        d_triangles, d_output,
        width, height,
        d_lights, d_light_colors, num_lights);
    // Copy back and free
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA error. %s\n", cudaGetErrorString(err));
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    // Free memory
    cudaFree(d_bbox);
    cudaFree(d_parent);
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_triangle);
    cudaFree(d_ray_origins);
    cudaFree(d_ray_directions);
    cudaFree(d_triangles);
    cudaFree(d_output);
    cudaFree(d_lights);
    cudaFree(d_light_colors);
    return h_output;
}
