#pragma once

#include "umm/session.h"

#include <filesystem>

namespace umm::cli {

void write_png(const std::filesystem::path & path, const umm::image_result & image);

}
