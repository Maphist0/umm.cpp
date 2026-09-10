// stb is a single-header library. Emit the implementation once in the CLI
// executable; the rest of the application uses this small wrapper.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image-io.h"

#include <stdexcept>

namespace umm::cli {

void write_png(const std::filesystem::path & path, const umm::image_result & image) {
    const auto expected_size = size_t(image.width) * image.height * 3;
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != expected_size) {
        throw std::runtime_error("Image result has invalid dimensions or pixel data");
    }
    if (!stbi_write_png(path.string().c_str(), image.width, image.height, 3,
                        image.rgb.data(), image.width * 3)) {
        throw std::runtime_error("Could not write image: " + path.string());
    }
}

}
