#ifndef LOAD_MESH_CUH
#define LOAD_MESH_CUH

#include "../shapes/triangle.cuh"
#include "../shapes/mesh.cuh"
#include "util.cuh"
#include <vector>  
#include <string>  
#include <sstream> 
#include <istream> 

class LoadMesh {
public:
    std::vector<std::variant<Triangle, Mesh>> objects;
    LoadMesh(const M4f& transform, std::istream& input_stream);
    Mesh get_mesh() const;
};

#endif // LOAD_MESH_CUH
