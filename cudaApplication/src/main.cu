#include "utils/main_util.cuh"

// Scene settings
int w = 224, h = 224 * 2;

// Camera settings
const float focal_length = 2.16;
const float field_of_view = 0.7854; // 45 degrees
const V3f camera_position(0, 0, -100); // -100 for car

// Rotation settings
bool rotate = true;
std::vector<V3f> rotations;

// Lights
std::vector<V3f> light_positions;
std::vector<V4f> light_colors;

// Meshes
std::vector<Mesh> meshes;

void setup_scene(int argc, char* argv[]){
    load_meshes(argc, argv, meshes);// Load all meshes

    rotations.emplace_back(-0.05, 0.4, 0.05);

    if (meshes.size() > 0 && rotate){
        for (auto r : rotations) meshes[0].triangles = rotate_mesh(meshes[0], r[0], r[1], r[2]); // Rotate mesh n
    }
    // Add light positions and colors
    light_colors.emplace_back(0.8f, 0.8f, 0.8f, 1.0f); // Light 1
    light_positions.emplace_back(0, 5, -30);

    light_colors.emplace_back(0.4f, 0.4f, 0.4f, 1.0f); // Light 2
    light_positions.emplace_back(10, -5, -20);

    light_colors.emplace_back(0.3f, 0.3f, 0.3f, 1.0f); // Light 3
    light_positions.emplace_back(10, 5, 20);

    light_colors.emplace_back(0.2f, 0.2f, 0.2f, 1.0f); // Light 4
    light_positions.emplace_back(-10, 20, -30);
}

int main(int argc, char* argv[])
{
    warmup_cuda();
    auto start = std::chrono::high_resolution_clock::now();
    setup_scene(argc, argv);
    float* output = h_raytrace(meshes, w, h, light_positions, light_colors, focal_length, field_of_view, camera_position);
    print_scene_in_ascii(output, w, h);
    std::cout << "Runtime: " << std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count() << " seconds" << std::endl;
    return 0;
}
