#include "main_util.cuh"

Mesh input_mesh(const std::string& filename) {
    std::ifstream file_stream;
    file_stream.open(filename);
    if (!file_stream) {
        std::cerr << "Can't open file: " << filename << std::endl;
        throw std::runtime_error("File opening failed.");
    }

    M4f identity_matrix;
    LoadMesh m(identity_matrix, file_stream);
    return m.get_mesh();
}

void load_meshes(int argc, char* argv[], std::vector<Mesh>& meshes) {
    for (int i = 1; i < argc; ++i) 
        meshes.push_back(input_mesh(argv[i])); // Load a single mesh with the LoadMesh class
}
