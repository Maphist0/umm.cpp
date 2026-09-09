#include "model-package.h"
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace umm {

model_paths resolve_model(const std::string & model, const std::string & generation) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(model)) {
        return {model, generation};
    }
    if (!generation.empty()) {
        throw std::invalid_argument("A model package cannot be combined with --image-model");
    }
    const auto root = fs::canonical(model);
    std::ifstream input(root / "model.json");
    if (!input) {
        throw std::runtime_error("Model package is missing model.json: " + model);
    }
    const auto manifest = nlohmann::json::parse(input);
    if (manifest.at("format") != "umm" || manifest.at("version") != 1 ||
        manifest.at("architecture") != "sensenova_u1") {
        throw std::runtime_error("Unsupported model package format, version, or architecture");
    }
    auto resolve = [&](const char * key) {
        const fs::path relative = manifest.at(key).get<std::string>();
        if (relative.empty() || relative.is_absolute()) {
            throw std::runtime_error(std::string("Invalid package path: ") + key);
        }
        const auto path = fs::weakly_canonical(root / relative);
        const auto within = path.lexically_relative(root);
        if (within.empty() || *within.begin() == ".." || !fs::is_regular_file(path)) {
            throw std::runtime_error(std::string("Missing or external package file: ") + key);
        }
        return path.string();
    };
    return {resolve("understanding"), resolve("generation")};
}

}
