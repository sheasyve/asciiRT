#include "util.hpp"

#include "ray.hpp"
#include "obj.hpp"
#include "triangle.hpp"
#include "sphere.hpp"
#include "mesh.hpp"

// Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; // 45 degrees
const bool is_perspective = true;
const Vector3d camera_position(0, 0, -10);

// Material settings
const Vector4d reflection_color(0.7, 0.7, 0.7, 0);
const Vector4d ambient_color(0.5, 0.5, 0.5, 0);
const Vector4d diffuse_color(0.5, 0.5, 0.5, 0);
const Vector4d specular_color(0.2, 0.2, 0.2, 0);
const double specular_exponent = 256.0;

// Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;
const Vector4d ambient_light(0.2, 0.2, 0.2, 0); // Ambient light

std::vector<std::shared_ptr<Intersectable>> objects;

std::optional<std::shared_ptr<Intersectable>> find_nearest_object(const Ray &ray)
{
    double min_t = std::numeric_limits<double>::infinity();
    std::optional<std::shared_ptr<Intersectable>> nearest_object = std::nullopt;

    for (const auto &object : objects)
    {
        auto res = object->intersects(ray);
        if (res.has_value())
        {
            double t = std::get<0>(res.value());
            if (t < min_t)
            {
                min_t = t;
                nearest_object = object;
            }
        }
    }

    return nearest_object;
}

Vector3d shoot_ray(const Ray &ray) {
    Vector3d color(0., 0., 0.);  // Initialize color

    // Get the nearest object (from the global `objects`)
    auto nearest_object_opt = find_nearest_object(ray);
    
    if (nearest_object_opt.has_value()) {
        auto nearest_object = nearest_object_opt.value();
        
        // Check if the nearest object is a Mesh
        if (auto mesh = std::dynamic_pointer_cast<Mesh>(nearest_object)) {
            std::cout << "Found a mesh for normal calculation." << std::endl;

            double min_t = std::numeric_limits<double>::infinity();  // Initialize min_t
            Vector3d p;  // Intersection point
            Eigen::Vector3d N;  // Normal vector

            // Loop through all triangles in the mesh
            for (const auto& triangle : mesh->triangles) {
                auto intersection = triangle->intersects(ray);  // Get intersection for each triangle
                if (intersection.has_value()) {
                    auto t = std::get<0>(intersection.value());
                    p = std::get<1>(intersection.value());

                    if (t < min_t) {  // Check if this is the closest intersection
                        min_t = t;  // Update min_t
                        N = triangle->normal();  // Get the normal of the triangle
                        std::cout << "Intersection found at distance: " << t 
                                  << " at point: " << p.transpose() << std::endl;
                        std::cout << "Normal: " << N.transpose() << std::endl;
                    }
                }
            }

            // Perform lighting calculations here if min_t was updated
            if (min_t < std::numeric_limits<double>::infinity()) {
                // Iterate through each light source
                for (const Vector3d &l_pos : light_positions) {
                    Vector3d L = (l_pos - p).normalized();  // Light direction
                    double dot_product = N.dot(L);  // Dot product for Lambertian reflection
                    double lambertian = std::max(dot_product, 0.0);  // Lambertian contribution
                    color += lambertian * Vector3d(1., 1., 1.);  // Assuming white light
                }
            }
        } else {
            std::cout << "Nearest object is not a mesh." << std::endl;
        }
    } else {
        std::cout << "No nearest object found." << std::endl;
    }

    return color;  // Return the calculated color
}


void print_scene_in_ascii(MatrixXd &Color, MatrixXd &A, int w, int h){
    // characters from https://stackoverflow.com/a/74186686
    const std::string brightness_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
    const int l = brightness_chars.size() - 1;
    for (int j = h-1; j >= 0; j--) {
        for (int i = w-1; i >= 0; i--) {
            char c = brightness_chars[(int)(l*Color(i,j))];
            std::cout << c;
        }
        std::cout << std::endl;
    }
}

void raytrace(int w, int h) {
    MatrixXd Color = MatrixXd::Zero(w, h);//Color of each pixel
    MatrixXd A = MatrixXd::Zero(w, h);  //Alpha value of each pixel

    const double aspect_ratio = double(w) / double(h);
    const double y = (((focal_length)*sin(field_of_view / 2)) / sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const double x = (y * aspect_ratio);
    const Vector3d image_origin(-x, y, camera_position[2] - focal_length);
    const Vector3d x_displacement(2.0 / w * x, 0, 0);
    const Vector3d y_displacement(0, -2.0 / h * y, 0);
    for (unsigned i = 0; i < w; ++i){
        for (unsigned j = 0; j < h; ++j){//Loop through every pixel
            const Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            const Ray r(camera_position, (camera_position - pixel_center).normalized());
            Vector3d result = shoot_ray(r);
            Color(i,j) = result(0,0);
        }
    }
    print_scene_in_ascii(Color, A, w, h);
}
//THis is what i have

void setup_scene()
{
    Obj o(Matrix4d::Identity());
    std::shared_ptr<Mesh> mesh = get_mesh(o);

    if (mesh && !mesh->triangles.empty())
    {
        std::cout << "Mesh loaded with " << mesh->triangles.size() << " triangles." << std::endl;
        objects.push_back(mesh);
    }
    else
    {
        std::cout << "Mesh loading failed or no triangles found." << std::endl;
    }

    // Light sample
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);
}

int main()
{
    int w = 200, h = 100;
    setup_scene();
    raytrace(w, h);
    return 0;
}
