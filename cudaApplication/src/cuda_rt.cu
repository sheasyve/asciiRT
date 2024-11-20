#include "cuda_rt.cuh"

__global__ void d_raytrace(
    V3f *ray_origins, V3f *ray_directions,
    AABB *nodes_bbox, int *nodes_left, int *nodes_right, int *nodes_triangle, int root_index,
    Triangle *triangles, float *output,
    int width, int height,
    V3f *light_positions,
    V4f *light_colors, int num_lights)
{
    // Ray initialization
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    V3f origin = ray_origins[idx];
    V3f direction = ray_directions[idx];
    // Shared memory allocation for lights
    extern __shared__ char shared_mem[];
    V3f *shared_light_positions = (V3f *)shared_mem;
    V4f *shared_light_colors = (V4f *)(shared_mem + num_lights * sizeof(V3f));
    // Load lights to shared memory
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    if (threadId < num_lights) {
        shared_light_positions[threadId] = light_positions[threadId];
        shared_light_colors[threadId] = light_colors[threadId];
    }
    __syncthreads();
    float brightness = 0.0;
    for (int depth = 0; depth < 3; depth++) { // Perform RT up to depth times for each reflection
        float local_brightness = 0.005, min_t = INF;
        int mindex = find_closest_triangle(origin, direction, nodes_bbox, nodes_left, nodes_right, nodes_triangle, root_index, triangles, min_t);
        if (mindex == -1) break;
        // Compute intersection point and normal
        V3f p = origin + direction * min_t;
        V3f N = triangles[mindex].normal().normalized();
        V3f V = (-direction).normalized();
        // Light calculations
        for (int i = 0; i < num_lights; i++) {
            V3f L = shared_light_positions[i] - p;
            float d = L.norm();
            L = L / d;
            V3f shadow_ray_origin = p + N * 1e-4f;// Shadow ray cast
            float shadow_ray_t = d; // Maximum distance to check (distance to light)
            int shadow_mindex = find_closest_triangle(shadow_ray_origin, L, nodes_bbox, nodes_left, nodes_right, nodes_triangle, root_index, triangles, shadow_ray_t);
            float attenuation = (shadow_mindex != -1 && shadow_ray_t > 0.0f) ? 0.0f : (1.0f / fmaf(0.01f, d * d, fmaf(0.1f, d, 1.0f))); // Shadow attenuation
            V3f light_rgb = V3f(shared_light_colors[i].x, shared_light_colors[i].y, shared_light_colors[i].z);
            local_brightness += attenuation * 0.4f * fmaxf(N.dot(L), 0.0f) * light_rgb.norm();// Diffuse lighting
            V3f R = (N * 2.0f * N.dot(L) - L).normalized();// Specular lighting
            local_brightness += attenuation * 0.4f * powf(fmaxf(R.dot(V), 0.0f), 32.0f) * light_rgb.norm();
        }
        brightness += powf(0.5f, depth) * fminf(local_brightness, 1.0f); 
        direction = direction - N * 2.0f * direction.dot(N);            
        direction = direction.normalized();
        origin = p + direction * 1e-4f;
    }
    output[idx] = brightness;
}

float *h_raytrace(
    std::vector<V3f> ray_origins,
    std::vector<V3f> ray_directions,
    std::vector<Mesh> meshes,
    int width, int height,
    std::vector<V3f> light_positions,
    std::vector<V4f> light_colors)
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
    std::cout << bvh.nodes.bbox[0].min[1] << " " << bvh.nodes.bbox[0].max[1];
 
    // Device pointers
    V3f *d_ray_origins = nullptr;
    V3f *d_ray_directions = nullptr;
    Triangle *d_triangles = nullptr;
    float *d_output = nullptr;
    V3f *d_lights = nullptr;
    V4f *d_light_colors = nullptr;
    AABB *d_bbox = nullptr;
    int *d_parent = nullptr;
    int *d_left = nullptr;
    int *d_right = nullptr;
    int *d_triangle = nullptr;

    // Allocate memory
    cudaMalloc((void **)&d_bbox, tree_size * sizeof(AABB));
    cudaMalloc((void **)&d_parent, tree_size * sizeof(int));
    cudaMalloc((void **)&d_left, tree_size * sizeof(int));
    cudaMalloc((void **)&d_right, tree_size * sizeof(int));
    cudaMalloc((void **)&d_triangle, tree_size * sizeof(int));
    cudaMalloc((void **)&d_ray_origins, size * sizeof(V3f));
    cudaMalloc((void **)&d_ray_directions, size * sizeof(V3f));
    cudaMalloc((void **)&d_triangles, num_triangles * sizeof(Triangle));
    cudaMalloc((void **)&d_output, size * sizeof(float));
    cudaMalloc((void **)&d_lights, num_lights * sizeof(V3f));
    cudaMalloc((void **)&d_light_colors, num_lights * sizeof(V4f));

    // Copy data to device
    cudaMemcpy(d_ray_origins, ray_origins.data(), size * sizeof(V3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_directions, ray_directions.data(), size * sizeof(V3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, light_positions.data(), num_lights * sizeof(V3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_light_colors, light_colors.data(), num_lights * sizeof(V4f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bbox, bvh.nodes.bbox, tree_size * sizeof(AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, bvh.nodes.parent, tree_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_left, bvh.nodes.left, tree_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, bvh.nodes.right, tree_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangle, bvh.nodes.triangle, tree_size * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel
    int minGridSize, blockSize = 16;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, d_raytrace, 0, 0);
    dim3 blockDim(blockSize);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    size_t smem_size = num_lights * (sizeof(V3f) + sizeof(V4f));
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
    return h_output;
}