#include "utils/main_util.hpp"

// Scene settings
int w = 224, h = 224*2;

// Camera settings
const float focal_length = 2.16;
const float field_of_view = 0.7854; // 45 degrees
const Eigen::Vector3f camera_position(0, 0, -100);

// Rotation settings
bool rotate = true;
float rX =-.05, rY =.4, rZ =.05;//Rotation IN RADIANS

// Lights
std::vector<Eigen::Vector3f> light_positions;
std::vector<Eigen::Vector4f> light_colors;

// Shader settings
const float diffuse_intensity = 0.4;
const float specular_intensity = 0.4;
const float ambient_light = 0.005;
const float shine = 32.0;
const float a = 1., b = .1, c = .01;//Attenuation constants
float reflection_coefficient = 0.5; 
// Variant to store different objects
using Intersectable = std::variant<Triangle, Sphere, Mesh>;
std::vector<Intersectable> objects;
std::vector<Mesh> meshes;

Eigen::Vector3f compute_normal(const std::variant<Triangle, Sphere, Mesh> &obj, const Eigen::Vector3f &hit_point, const Triangle *hit_triangle = nullptr){
    return std::visit([&](const auto &shape) -> Eigen::Vector3f{
        if constexpr (std::is_same_v<decltype(shape), const Sphere&>) {//Sphere
            return (hit_point - shape.center).normalized();
        } else if constexpr (std::is_same_v<decltype(shape), const Triangle&>) {//Triangle
            return shape.normal();
        } else if constexpr (std::is_same_v<decltype(shape), const Mesh&>) {//Mesh
            if (hit_triangle != nullptr) {
                return hit_triangle->normal();  
            }
            return Eigen::Vector3f(0, 0, 0);  // No valid hit triangle, fallback
        }
    }, obj);
}

std::optional<std::tuple<float, Eigen::Vector3f, Intersectable, const Triangle *>> find_nearest_object(const Ray &ray){
    //Find the nearest intersecting mesh or sphere
    std::optional<std::tuple<float, Eigen::Vector3f, Intersectable, const Triangle *>> nearest_hit;
    float min_t = INFINITY;
    for (const auto &object : objects){
        auto hit = std::visit(Intersect(ray), object);//Find intersection based on this object type
        if (hit.has_value()){
            float t = std::get<0>(hit.value());
            if (t < min_t){
                min_t = t;
                const Triangle *hit_triangle = nullptr;
                if (std::holds_alternative<Mesh>(object)){//Mesh
                    const Mesh &mesh = std::get<Mesh>(object);
                    auto mesh_hit = mesh.intersects(ray);
                    if (mesh_hit.has_value()) hit_triangle = std::get<2>(mesh_hit.value()); // Get the hit triangle
                }
                // For Sphere, hit_triangle remains nullptr.
                nearest_hit = std::make_tuple(t, std::get<1>(hit.value()), object, hit_triangle);
            }
        }
    }
    return nearest_hit;
}

Eigen::Vector3f shoot_ray(const Ray &ray, int depth = 0) {
    Eigen::Vector3f color(0., 0., 0.); 
    if (depth > 3) return color; // Max reflection depth
    auto nearest_hit = find_nearest_object(ray);
    if (nearest_hit.has_value()) {
        float brightness = ambient_light;
        const auto &[t, hit_point, nearest_object, hit_triangle] = nearest_hit.value();
        Eigen::Vector3f N = compute_normal(nearest_object, hit_point, hit_triangle);
        N.normalize();
        Eigen::Vector3f V = -ray.direction;
        V.normalize();
        
        for (int i = 0; i < light_positions.size(); i++) {
            Eigen::Vector3f L = (light_positions[i] - hit_point);
            float d = L.norm();
            L.normalize();
            
            // Shadow ray
            Ray shadow_ray(hit_point + N * 1e-4, L); // Offset to avoid self-intersection
            auto shadow_hit = find_nearest_object(shadow_ray);
            bool in_shadow = false;
            
            if (shadow_hit.has_value()) {
                const auto &[sh_t, sh_hit_point, sh_nearest_object, sh_hit_triangle] = shadow_hit.value();
                in_shadow = sh_t < d; // Only in shadow if the hit is closer than the light source
            }
            
            Eigen::Vector3f light_rgb = light_colors[i].head<3>();
            float attenuation = in_shadow ? 0.0 : 1.0 / (a + b * d + c * d * d);
            
            // Diffuse shading
            float lambertian = fmax(N.dot(L), 0.0);
            brightness += attenuation * diffuse_intensity * lambertian * light_rgb.norm();
            
            // Specular shading
            Eigen::Vector3f R = (2.0 * N.dot(L) * N - L).normalized();
            float spec_angle = fmax(R.dot(V), 0.0);
            float specular = pow(spec_angle, shine);
            brightness += attenuation * specular_intensity * specular * light_rgb.norm();
        }

        brightness = fmin(brightness, 1.0);

        // Recursive reflection
        Eigen::Vector3f R = (ray.direction - 2.0 * ray.direction.dot(N) * N).normalized();
        Ray reflected_ray(hit_point + R * 1e-4, R);
        Eigen::Vector3f reflected_color = shoot_ray(reflected_ray, depth + 1);
        
        color = Eigen::Vector3f(brightness, brightness, brightness) 
              + reflection_coefficient * reflected_color;
    }
    return color;
}

void print_scene_in_ascii(const Eigen::MatrixXf &Color, int w, int h) {
    // ASCII characters for brightness levels
    const std::string brightness_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
    const int l = brightness_chars.size() - 1;
    auto [first_line, last_line] = find_boundary(Color, w, h);
    for (int j = first_line; j >= last_line; --j) {
        for (int i = 0; i < w; ++i) {
            double brightness = Color(i, j);
            brightness = std::max(0.0, std::min(1.0, brightness)); 
            char c = brightness_chars[static_cast<int>(l * brightness)];
            std::cout << c;
        }
        std::cout << std::endl;
    }
}

void raytrace(int w, int h){
    Eigen::MatrixXf Color = Eigen::MatrixXf::Zero(w, h); 
    const float aspect_ratio = float(w) / float(h);
    const float y = (((focal_length)*sin(field_of_view / 2)) / sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const float x = (y * aspect_ratio);
    Eigen::Vector3f image_origin(-x, y, camera_position[2] - focal_length);
    Eigen::Vector3f x_displacement(2.0 / w * x, 0, 0);
    Eigen::Vector3f y_displacement(0, -2.0 / h * y, 0);
    for (unsigned i = 0; i < w; ++i){
        for (unsigned j = 0; j < h; ++j){
            Eigen::Vector3f pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            Ray r(camera_position, (camera_position - pixel_center).normalized());
            Eigen::Vector3f result = shoot_ray(r);
            if (result.size() > 0){
                Color(i, j) = result(0);
            }
            else{
                std::cerr << "Invalid ray result." << std::endl;
            }
        }
    }
    print_scene_in_ascii(Color, w, h);
}

void setup_scene(int argc, char* argv[]){ 
    load_meshes(argc,argv,meshes);
    float rX =-.05, rY =.4, rZ =.05;//Rotation IN RADIANS
    if(meshes.size() > 0 && rotate) meshes[0].triangles = rotate_mesh(meshes[0],rX,rY,rZ);//Rotate mesh 1
    //meshes[0].triangles = translate_mesh(meshes[0],5,5,5);//Translate mesh 1
    for (auto &mesh : meshes) objects.emplace_back(mesh);

    //Sphere example
    //Eigen::Vector3f sphere_center(0, 0, 1);               
    //objects.emplace_back(Sphere(sphere_center, 1.)); 
               
    light_colors.emplace_back(0.8, 0.8, 0.8, 1);//Light 1
    light_positions.emplace_back(0, 5, -30);  
    light_colors.emplace_back(0.4, 0.4, 0.4, 1);//Light 2
    light_positions.emplace_back(10, -5, -20);  
    light_colors.emplace_back(0.3, 0.3, 0.3, 1);//Light 3  
    light_positions.emplace_back(10, 5, 20);  
    light_colors.emplace_back(0.2, 0.2, 0.2, 1);//Light 4  
    light_positions.emplace_back(-10, 20, -30); 
}

int main(int argc, char* argv[]){
    auto start=std::chrono::high_resolution_clock::now();
    setup_scene(argc,argv);
    raytrace(w, h);
    auto end=std::chrono::high_resolution_clock::now();
    std::cout<<"Runtime: "<<std::chrono::duration<float>(end-start).count()<<" seconds"<<std::endl;
    return 0;
}
