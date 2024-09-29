#include "util.hpp"

using namespace Eigen;//Eigen library for linear algebra, matrix handling

//Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; //45 degrees
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 2);

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

bool ray_triangle_intersection(){}//Code to intersect a ray with a triangle, requires building aabb tree

bool ray_sphere_intersection(){}//Code to intersect a ray with a sphere

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

bool find_nearest_object(const Vector3d &ray_origin, const Vector3d &ray_direction, Vector3d &p, Vector3d &N){
//TODO: Implement code to find the nearest object that intersects with the shot ray, or false if no object is found
}

Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction){
    Vector3d p, N;
    const bool nearest_object = find_nearest_object(ray_origin, ray_direction, p, N);
    //TODO: Shoot ray, check intersections and return the color after processing each light source
}

void print_scene_in_ascii(MatrixXd &Color, MatrixXd &A){
    //#TODO: Print the scene in ASCII from the matrix of colors and matrix of alpha values

}

void raytrace(int w, int h) {
    //Code to generate color values for each pixel
    //TODO: Implement raytracing algorithm
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
            //TODO: Do raytracing here for this ray
        }
    }
    print_scene_in_ascii(Color,A);
}

void setup_scene(){
    //TODO: Add objects to the scene and set up environment

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
