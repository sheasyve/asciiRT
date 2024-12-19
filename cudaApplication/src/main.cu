#include "utils/main_util.cuh"

// Set up for an animation with car 

// Scene settings
int w = 192, h = 128;

// Camera settings
const float focal_length = 2.16;
const float field_of_view = 0.7854; // 45 degrees
const V3f camera_position(0., .75, -45); 

// Animation settings
bool animate = true;
constexpr int TARGET_FPS = 60;
constexpr int FRAME_TIME_MS = 1000 / TARGET_FPS;

// Rotation and translation settings
int mesh_count = 1;
bool rotate = true;
bool translate = true;
std::vector<V3f> rotations;
std::vector<V3f> translations;
std::vector<V3f> rotation_increments;

// Lights
std::vector<V3f> light_positions;
std::vector<V4f> light_colors;

// Meshes
std::vector<Mesh> meshes;

void setup_scene(int argc, char* argv[]){
    load_meshes(argc, argv, meshes); // Load all meshes
    rotations.emplace_back(-0.05, 0.4, 0.05);
    translations.emplace_back(4.,5.,0.);
    rotation_increments.emplace_back(0., 0.05, 0.);
    if (!meshes.empty() > 0 && rotate) {
        for (auto r : rotations) meshes[0].triangles = rotate_mesh(meshes[0], r[0], r[1], r[2]);
    }
    if (!meshes.empty() && translate){
        for (auto t: translations) meshes[0].triangles = translate_mesh(meshes[0], t[0], t[1], t[2]);
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

void gen_rays(int w, int h, std::vector<V3f> &ray_origins, std::vector<V3f> &ray_directions, float focal_length, float field_of_view, V3f camera_position)
{
    const float aspect_ratio = float(w) / float(h);
    const float y = (((focal_length)*sin(field_of_view / 2)) / sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const float x = (y * aspect_ratio);
    V3f image_origin(-x, y, camera_position[2] - focal_length);
    V3f x_displacement(2.0 / w * x, 0, 0);
    V3f y_displacement(0, -2.0 / h * y, 0);
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            V3f pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            ray_origins.push_back(camera_position);
            ray_directions.push_back((camera_position - pixel_center).normalized());
        }
    }
}

void animate_main(std::vector<V3f> ray_origins, std::vector<V3f> ray_directions) {
    while (1) {
        auto start = std::chrono::high_resolution_clock::now();
        if (rotate && !meshes.empty()) {
            for (int i = 0; i < mesh_count; i++) {
                meshes[i].triangles = rotate_mesh(meshes[i], rotation_increments[i][0], rotation_increments[i][1], rotation_increments[i][2]);
            }
        }
        std::vector<Triangle> triangles = get_triangles(meshes);
        BvhTree bvh(triangles);
        float* output = h_raytrace(triangles, w, h, light_positions, light_colors, focal_length, field_of_view, camera_position, bvh, ray_origins, ray_directions);
        std::cout << "\033[H\033[J" << std::flush;
        print_scene_in_ascii(output, w, h);
        delete[] output;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (elapsed_time < FRAME_TIME_MS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_TIME_MS - elapsed_time));
        }
    }
}

void photo_main(std::vector<V3f> ray_origins, std::vector<V3f> ray_directions){
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Triangle> triangles = get_triangles(meshes);
    BvhTree bvh(triangles);
    float* output = h_raytrace(triangles, w, h, light_positions, light_colors, focal_length, field_of_view, camera_position, bvh, ray_origins, ray_directions);
    std::cout << "\033[H\033[J"; 
    print_scene_in_ascii(output, w, h);
    delete[] output;
    std::cout << "Runtime: " << std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count() << " seconds" << std::endl;
}

int main(int argc, char* argv[])
{
    warmup_cuda();
    setup_scene(argc, argv);
    mesh_count = argc - 1;
    std::vector<V3f> ray_origins, ray_directions;
    gen_rays(w, h, ray_origins, ray_directions, focal_length, field_of_view, camera_position);
    if(animate){
        animate_main(ray_origins, ray_directions);
        return 0;
    }
    photo_main(ray_origins, ray_directions);
    return 0;
}
