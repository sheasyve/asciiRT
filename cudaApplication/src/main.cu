#include "utils/main_util.cuh"

// Scene settings
int w = 224, h = 224 * 2;

// Camera settings
const float focal_length = 2.16;
const float field_of_view = 0.7854; // 45 degrees
const V3f camera_position(0, 0, -100); // -100 for car

// Rotation settings
bool rotate = true;

// Lights
std::vector<V3f> light_positions;
std::vector<V4f> light_colors;

// Meshes
std::vector<Mesh> meshes;

void gen_rays(int w, int h, std::vector<V3f>& ray_origins, std::vector<V3f>& ray_directions)
{
    const float aspect_ratio = float(w) / float(h);
    const float y = (((focal_length) * sin(field_of_view / 2)) / 
                     sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const float x = (y * aspect_ratio);

    V3f image_origin(-x, y, camera_position.z - focal_length);
    V3f x_displacement(2.0 / w * x, 0, 0);
    V3f y_displacement(0, -2.0 / h * y, 0);

    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            V3f pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            ray_origins.push_back(camera_position);
            ray_directions.push_back((pixel_center - camera_position).normalized());
        }
    }
    std::cout << "Generated " << ray_origins.size() << " rays.\n";
    std::cout << "First ray origin: (" << ray_origins[0].x << ", " << ray_origins[0].y << ", " << ray_origins[0].z << ")\n";
    std::cout << "First ray direction: (" << ray_directions[0].x << ", " << ray_directions[0].y << ", " << ray_directions[0].z << ")\n";
}

void setup_scene(int argc, char* argv[])
{
    load_meshes(argc, argv, meshes);

    float rX = -0.05, rY = 0.4, rZ = 0.05; // Rotation in radians
    if (meshes.size() > 0 && rotate)
        meshes[0].triangles = rotate_mesh(meshes[0], rX, rY, rZ); // Rotate mesh 1

    // Add light positions and colors
    light_colors.emplace_back(0.8f, 0.8f, 0.8f, 1.0f); // Light 1
    light_positions.emplace_back(0, 5, -30);

    light_colors.emplace_back(0.4f, 0.4f, 0.4f, 1.0f); // Light 2
    light_positions.emplace_back(10, -5, -20);

    light_colors.emplace_back(0.3f, 0.3f, 0.3f, 1.0f); // Light 3
    light_positions.emplace_back(10, 5, 20);

    light_colors.emplace_back(0.2f, 0.2f, 0.2f, 1.0f); // Light 4
    light_positions.emplace_back(-10, 20, -30);
    std::cout << "Number of meshes loaded: " << meshes.size() << "\n";
    for (size_t i = 0; i < meshes.size(); ++i) {
        std::cout << "Mesh " << i << " has " << meshes[i].triangles.size() << " triangles.\n";
    }

}

int main(int argc, char* argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<V3f> ray_origins, ray_directions;
    gen_rays(w, h, ray_origins, ray_directions);
    setup_scene(argc, argv);
    float* output = h_raytrace(ray_origins, ray_directions, meshes, w, h, light_positions, light_colors);
    print_scene_in_ascii(output, w, h);
    std::cout << "Runtime: " << std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count() << " seconds" << std::endl;

    return 0;
}
