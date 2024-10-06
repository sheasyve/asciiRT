#include "util.hpp"

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