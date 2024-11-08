#include "utils/main_util.hpp"

// Scene settings
int w = 112*2, h = 224*2;

// Camera settings
const float focal_length = 2.16;
const float field_of_view = 0.7854; // 45 degrees
const Eigen::Vector3f camera_position(0, 0, -100);

// Rotation settings
bool rotate = false;
float rX =-.05, rY =.4, rZ =.05;//Rotation IN RADIANS

// Lights
std::vector<Eigen::Vector3f> light_positions;
std::vector<Eigen::Vector4f> light_colors;
int num_lights = 4;

// Shader settings
const float diffuse_intensity = 0.4;
const float specular_intensity = 0.4;
const float ambient_light = 0.005;
const float shine = 32.0;
const float a = 1., b = .1, c = .01;//Attenuation constants

// Variant to store different objects
std::vector<Mesh> meshes;

void d_raytrace(
    Eigen::Vector3f ray_origin, Eigen::Vector3f ray_direction, 
    std::vector<BvhTree::Node>& nodes, int root_index, std::vector<Triangle> triangles,
    float* output,
    int width, int height, int x, int y
) {
    //Ray init
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    Eigen::Vector3f origin = ray_origin;
    Eigen::Vector3f direction = ray_direction;
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

void gen_rays(int w, int h, std::vector<Eigen::Vector3f> &ray_origins, std::vector<Eigen::Vector3f> &ray_directions, float* output)
{
    int size = w * h;
    int num_lights = static_cast<int>(light_positions.size());
    std::vector<Triangle> triangles = get_triangles(meshes);
    int num_triangles = static_cast<int>(triangles.size());
    // Build BVH
    BvhTree bvh(triangles);
    std::vector<BvhTree::Node>& nodes = bvh.nodes;
    int tree_size = static_cast<int>(nodes.size());
    const float aspect_ratio = float(w) / float(h);
    const float y = (((focal_length)*sin(field_of_view / 2)) / sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const float x = (y * aspect_ratio);
    Eigen::Vector3f image_origin(-x, y, camera_position[2] - focal_length);
    Eigen::Vector3f x_displacement(2.0 / w * x, 0, 0);
    Eigen::Vector3f y_displacement(0, -2.0 / h * y, 0);
    int root = bvh.root;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            Eigen::Vector3f pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            Eigen::Vector3f ray_origin = camera_position;
            Eigen::Vector3f ray_direction = (camera_position - pixel_center).normalized();
            d_raytrace(ray_origin, ray_direction, nodes, root, triangles, output, w, h, i, j);
        }
    }
}

void setup_scene(int argc, char* argv[]){ 
    load_meshes(argc, argv, meshes);
    float rX = -.05, rY = .8, rZ = .05; // Rotation IN RADIANS
    if (meshes.size() > 0 && rotate)
        meshes[0].triangles = rotate_mesh(meshes[0], rX, rY, rZ); // Rotate mesh 1
    // meshes[0].triangles = translate_mesh(meshes[0],5,5,5);//Translate mesh 1
    light_colors.emplace_back(0.8, 0.8, 0.8, 1); // Light 1
    light_positions.emplace_back(0, 5, -30);
    light_colors.emplace_back(0.4, 0.4, 0.4, 1); // Light 2
    light_positions.emplace_back(10, -5, -20);
    light_colors.emplace_back(0.3, 0.3, 0.3, 1); // Light 3
    light_positions.emplace_back(10, 5, 20);
    light_colors.emplace_back(0.2, 0.2, 0.2, 1); // Light 4
    light_positions.emplace_back(-10, 20, -30);
}

int main(int argc, char* argv[]){
    auto start=std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector3f> ray_origins, ray_directions;
    setup_scene(argc,argv);
    float* output = new float[w * h];
    gen_rays(w, h, ray_origins, ray_directions, output);
    print_scene_in_ascii(output, w, h);
    auto end=std::chrono::high_resolution_clock::now();
    std::cout<<"Runtime: "<<std::chrono::duration<float>(end-start).count()<<" seconds"<<std::endl;
    delete[] output;
    return 0;
}
