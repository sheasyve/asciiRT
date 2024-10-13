#include "utils/main_util.cuh"
#include "cuda_rt.cuh"

// Camera settings
const double focal_length = 2.16;
const double field_of_view = 0.7854; // 45 degrees
const Eigen::Vector3d camera_position(0, 0, -100);

// Lights
std::vector<Eigen::Vector3d> light_positions;
std::vector<Eigen::Vector4d> light_colors;

// Variant to store different objects
using Intersectable = std::variant<Triangle, Sphere, Mesh>;
std::vector<Intersectable> objects;

Eigen::Vector3d compute_normal(const std::variant<Triangle, Sphere, Mesh> &obj, const Eigen::Vector3d &hit_point, const Triangle *hit_triangle = nullptr){
    return std::visit([&](const auto &shape) -> Eigen::Vector3d{
        if constexpr (std::is_same_v<decltype(shape), const Sphere&>) {//Sphere
            return (hit_point - shape.center).normalized();
        } else if constexpr (std::is_same_v<decltype(shape), const Triangle&>) {//Triangle
            return shape.normal();
        } else if constexpr (std::is_same_v<decltype(shape), const Mesh&>) {//Mesh
            if (hit_triangle != nullptr) {
                return hit_triangle->normal();  
            }
            return Eigen::Vector3d(0, 0, 0);  // No valid hit triangle, fallback
        }
    }, obj);
}

void print_scene_in_ascii(double* color, int w, int h) {
    // ASCII characters for brightness levels
    const std::string brightness_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
    const int l = brightness_chars.size() - 1;
    auto [first_line, last_line] = find_boundary(color, w, h);
    for (int j = first_line; j >= last_line; --j) {
        for (int i = 0; i < w; ++i) {
            double brightness = color[j*w + i];
            brightness = std::max(0.0, std::min(1.0, brightness)); // Clamp brightness between 0 and 1
            char c = brightness_chars[static_cast<int>(l * brightness)];
            std::cout << c;
        }
        std::cout << std::endl;
    }
}

void setup_scene(){ 
    light_colors.emplace_back(0.8, 0.8, 0.8, 1);  
    light_positions.emplace_back(0, 5, -30);  

    light_colors.emplace_back(0.4, 0.4, 0.4, 1); 
    light_positions.emplace_back(10, -5, -20);  

    light_colors.emplace_back(0.3, 0.3, 0.3, 1);  
    light_positions.emplace_back(10, 5, 20);  

    light_colors.emplace_back(0.2, 0.2, 0.2, 1);  
    light_positions.emplace_back(-10, 20, -30);  
}

std::vector<Ray> gen_rays(int w, int h) {
    std::vector<Ray> rays;
    const double aspect_ratio = double(w) / double(h);
    const double y = (((focal_length)*sin(field_of_view / 2)) / sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const double x = (y * aspect_ratio);
    Eigen::Vector3d image_origin(-x, y, camera_position[2] - focal_length);
    Eigen::Vector3d x_displacement(2.0 / w * x, 0, 0);
    Eigen::Vector3d y_displacement(0, -2.0 / h * y, 0);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            Eigen::Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            Ray r(camera_position, (camera_position - pixel_center).normalized());
            rays.push_back(r);
        }
    }
    return rays;
}

int main(int argc, char* argv[]){
    std::istream *input_stream = nullptr;
    std::ifstream file_stream;
    if (argc >= 2) {
        std::string obj_file_path = argv[1];
        file_stream.open(obj_file_path);
        if (!file_stream) {
            std::cerr << "can't open file" << std::endl;
            return 1;
        }
        input_stream = &file_stream; 
    } else {
        input_stream = &std::cin;  
    }
    
    setup_scene();
    LoadMesh m(Eigen::Matrix4d::Identity(), *input_stream);
    Mesh mesh = m.get_mesh();
    int w = 112*2, h = 224*2;
    
    //Rotation IN RADIANS
    double rX =-.05, rY =.4, rZ =.05;
    double* output = h_raytrace(&gen_rays(w, h)[0], mesh, w, h, light_positions,light_colors,rX,rY,rZ);//Cuda Kernel
    print_scene_in_ascii(output, w, h);
    return 0;
}
