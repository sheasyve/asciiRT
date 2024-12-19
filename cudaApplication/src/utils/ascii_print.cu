#include "ascii_print.cuh"

void print_scene_in_ascii(float* color, int w, int h) {
    // ASCII characters for brightness levels
    const std::string brightness_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
    const int l = brightness_chars.size() - 1;
    // Print the model in ASCII
    for (int j = h; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            double brightness = color[j * w + i];
            brightness = std::max(0.0, std::min(1.0, brightness)); // Clamp brightness between 0 and 1
            char c = brightness_chars[static_cast<int>(l * brightness)];
            std::cout << c;
        }
        std::cout << "\n";
    }
}
