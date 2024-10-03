#include "util.hpp"
#include "ray.hpp"
#include "obj.hpp"
#include "triangle.hpp"

using namespace Eigen;//Eigen library for linear algebra, matrix handling

//Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; //45 degrees
const bool is_perspective = true;
const Vector3d camera_position(0, 0, -10);

//Material settings
const Vector4d reflection_color(0.7, 0.7, 0.7, 0);
const Vector4d ambient_color(0.5, 0.5, 0.5, 0);
const Vector4d diffuse_color(0.5, 0.5, 0.5, 0);
const Vector4d specular_color(0.2, 0.2, 0.2, 0);
const double specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;
const Vector4d ambient_light(0.2, 0.2, 0.2, 0);//Ambient light

std::vector<Triangle> triangles;

//Code to intersect a ray with a box
bool ray_box_intersection(const Vector3d& ray_origin, const Vector3d& ray_direction, const AlignedBox3d& box){
    Vector3d inv_raydir = 1 / ray_direction.array();
    double tmin = 0, tmax = INFINITY;
    for (int d = 0; d < 3; d++) {
        const double t1 = (box.min()(d) - ray_origin(d)) * inv_raydir(d);
        const double t2 = (box.max()(d) - ray_origin(d)) * inv_raydir(d);
        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));
    }
    return tmin < tmax;
}

std::optional<Triangle> find_nearest_object(const Ray ray){
    double min_t = 1000000;
    std::optional<Triangle> min_tri = {};
    for (auto tri : triangles) {
        auto res = tri.intersects(ray);
        if (res.has_value()) {
            auto tri_tup = res.value();
            auto t = std::get<0>(tri_tup);
            if (t < min_t) {
                min_t = t;
                min_tri = tri;
            }
        }
    }
    return min_tri;
}

Vector3d shoot_ray(const Ray ray){
    Vector3d color(0.,0.,0.);

    const std::optional<Triangle> nearest_object_opt = find_nearest_object(ray);
    
    if (nearest_object_opt.has_value()) {
        auto nearest = nearest_object_opt.value();
        std::tuple<double, Vector3d> tup = nearest.intersects(ray).value();
        
        auto t = std::get<0>(tup);
        auto p = std::get<1>(tup);
        auto N = nearest.normal();

        for (Vector3d l_pos : light_positions) {
            Vector3d L = (l_pos - p).normalized();
            float lambertian = std::max(N.dot(L), 0.0);
            color += lambertian*Vector3d(1.,1.,1.);
        }
    }
    return color;
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
            const Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;//Not sure if this pixel offset is right for ascii terminal out
            const Ray r(camera_position, (camera_position - pixel_center).normalized());
            Vector3d result = shoot_ray(r);
            Color(i,j) = result(0,0);
        }
    }
    print_scene_in_ascii(Color, A, w, h);
}

void setup_scene(){
    //TODO: Add objects to the scene and set up environment
    Obj o(Matrix4d::Identity());
    triangles = o.triangles;
    //Light sample
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

}

int main() {
    int w= 200, h = 200;
    setup_scene();
    raytrace(w, h);
    
    return 0;
}
