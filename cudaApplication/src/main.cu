#include "utils/main_util.cuh"

// Scene settings
int w = 112*2, h = 224*2;

// Camera settings
const double focal_length = 2.16;
const double field_of_view = 0.7854; // 45 degrees
const Eigen::Vector3d camera_position(0, 0, -100);

// Rotation settings
bool rotate = true;

// Lights
std::vector<Eigen::Vector3d> light_positions;
std::vector<Eigen::Vector4d> light_colors;

// Meshes
std::vector<Mesh> meshes;

void gen_rays(int w, int h, std::vector<Eigen::Vector3d>& ray_origins, std::vector<Eigen::Vector3d>& ray_directions) {
    const double aspect_ratio = double(w) / double(h);
    const double y = (((focal_length)*sin(field_of_view / 2)) / sin((180 - (90 + ((field_of_view * (180 / M_PI) / 2)))) * (M_PI / 180)));
    const double x = (y * aspect_ratio);
    Eigen::Vector3d image_origin(-x, y, camera_position[2] - focal_length);
    Eigen::Vector3d x_displacement(2.0 / w * x, 0, 0);
    Eigen::Vector3d y_displacement(0, -2.0 / h * y, 0);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            Eigen::Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;
            ray_origins.push_back(camera_position);
            ray_directions.push_back((camera_position - pixel_center).normalized());
        }
    }
}

void setup_scene(int argc, char* argv[]){ 
    load_meshes(argc,argv,meshes);
    double rX =-.05, rY =.4, rZ =.05;//Rotation IN RADIANS
    if(meshes.size() > 0 && rotate) meshes[0].triangles = rotate_mesh(meshes[0],rX,rY,rZ);//Rotate mesh 1
    //meshes[0].triangles = translate_mesh(meshes[0],5,5,5);//Translate mesh 1
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
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector3d> ray_origins, ray_directions;
    gen_rays(w,h,ray_origins,ray_directions);
    setup_scene(argc,argv);
    double* output = h_raytrace(ray_origins, ray_directions, meshes, w, h, light_positions, light_colors);
    print_scene_in_ascii(output,w,h);
    std::cout << "Runtime: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-start).count() << " seconds" <<std::endl;
    return 0;
}
