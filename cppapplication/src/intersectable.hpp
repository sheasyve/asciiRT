#ifndef INTERSECTABLE_HPP
#define INTERSECTABLE_HPP

#include <optional>
#include <tuple>
#include <Eigen/Dense>
using namespace Eigen;

class Ray;//Forward Declaration

class Intersectable {
public:
    virtual std::optional<std::tuple<double, Eigen::Vector3d>> intersects(const Ray& ray) const = 0;
    virtual ~Intersectable() = default;
};

#endif
