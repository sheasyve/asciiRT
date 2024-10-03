#include "triangle.hpp"
#include "obj.hpp"

Obj::Obj(Matrix4d transform) {
    std::vector<Vector4d> points;
    
    points.push_back(Vector4d(0.,0.,0.,0.)); // .obj is one-indexed
    for (std::string line; std::getline(std::cin, line);) {
        
        std::stringstream line_stream(line);

        std::string s;
        float x;
        float y;
        float z;
        line_stream >> s >> x >> y >> z;
        switch (line[0]) {
            case 'v':
                points.push_back(transform*Vector4d(x, y, z, 1.));
                break;
            case 'f':
                int ix = x;
                int iy = y;
                int iz = z;
                triangles.push_back(Triangle(points[ix].head<3>(), points[iy].head<3>(), points[iz].head<3>()));
                break;
        }
    }

    std::cout << triangles.size() << std::endl;
}