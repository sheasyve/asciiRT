#include "utils/main_util.hpp"

// Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; // 45 degrees
const Eigen::Vector3d camera_position(0, 0, -10);

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

std::optional<std::tuple<double, Eigen::Vector3d, Intersectable, const Triangle *>> find_nearest_object(const Ray &ray){
    //Find the nearest intersecting mesh or sphere
    std::optional<std::tuple<double, Eigen::Vector3d, Intersectable, const Triangle *>> nearest_hit;
    double min_t = INFINITY;
    for (const auto &object : objects){
        auto hit = std::visit(Intersect(ray), object);//Find intersection based on this object type
        if (hit.has_value()){
            double t = std::get<0>(hit.value());
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


Eigen::Vector3d shoot_ray(const Ray &ray){
    Eigen::Vector3d color(0., 0., 0.);
    auto nearest_hit = find_nearest_object(ray);
    if (nearest_hit.has_value()){
        const auto &[t, hit_point, nearest_object, hit_triangle] = nearest_hit.value();
        Eigen::Vector3d normal = compute_normal(nearest_object, hit_point, hit_triangle);
        // Lighting calculations
        for (const Eigen::Vector3d &light_pos : light_positions){
            Eigen::Vector3d L = (light_pos - hit_point).normalized();
            double lambertian = std::max(normal.dot(L), 0.0);
            color += lambertian * Eigen::Vector3d(1., 1., 1.);
        }
    }
    return color;
}

void raytrace(int w, int h){
    Eigen::MatrixXd Color = Eigen::MatrixXd::Zero(w, h); 
    const double aspect_ratio = double(w) / double(h);
    const double y = (((focal_length)*sin(field_of_view / 2)) / sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const double x = (y * aspect_ratio);
    Eigen::Vector3d image_origin(-x, y, camera_position[2] - focal_length);
    Eigen::Vector3d x_displacement(2.0 / w * x, 0, 0);
    Eigen::Vector3d y_displacement(0, -2.0 / h * y, 0);
    for (unsigned i = 0; i < w; ++i){
        for (unsigned j = 0; j < h; ++j){
            Eigen::Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            Ray r(camera_position, (camera_position - pixel_center).normalized());
            Eigen::Vector3d result = shoot_ray(r);
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

void setup_scene(){
    Obj o(Eigen::Matrix4d::Identity()); //Mesh
    Mesh mesh = o.get_mesh();           
    if (!mesh.triangles.empty()) objects.emplace_back(mesh);// Add mesh to objects
    //Sphere example
    //Eigen::Vector3d sphere_center(0, 0, 1);               
    //objects.emplace_back(Sphere(sphere_center, 1.));            
    light_positions.emplace_back(8, 8, 0);  // Light position
    light_colors.emplace_back(16, 16, 16, 0);  // Light intensity
}

void print_scene_in_ascii(const Eigen::MatrixXd &Color, int w, int h) {
    // ASCII characters for brightness levels
    const std::string brightness_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
    const int l = brightness_chars.size() - 1;
    auto [first_line, last_line] = find_boundary(Color, w, h);
    for (int j = first_line; j >= last_line; --j) {
        for (int i = 0; i < w; ++i) {
            double brightness = Color(i, j);
            brightness = std::max(0.0, std::min(1.0, brightness)); // Clamp brightness between 0 and 1
            char c = brightness_chars[static_cast<int>(l * brightness)];
            std::cout << c;
        }
        std::cout << std::endl;
    }
}

int main(){
    int w = 202, h = 113;
    setup_scene();
    raytrace(w, h);
    return 0;
}
