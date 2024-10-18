#include "ascii_print.hpp"

std::pair<int, int> find_boundary(double* color, int w, int h) {
    //Find the first and last non-empty lines to print without extra whitespace
    int first_line = -1,last_line = -1;
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            if (color[j*w + i] > 0.0) {
                if (first_line == -1) {
                    first_line = j; 
                }
                last_line = j;  
                break;
            }
        }
    }
    return {first_line, last_line}; 
}

void print_scene_in_ascii(double* color, int w, int h) {
    // ASCII characters for brightness levels
    const std::string brightness_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
    const int l = brightness_chars.size() - 1;
    auto [first_line, last_line] = find_boundary(color, w, h);
    for (int j = first_line; j >= last_line; --j) {
        for (int i = 0; i < w; ++i) {
            double brightness = color[j*w + i];
            brightness = std::max(0.0, std::min(1.0, brightness)); // Clamp brightness between 0 and 1
            char c = brightness_chars[static_cast<int>(l * brightness)];
            std::cout << c;
        }
        std::cout << std::endl;
    }
}