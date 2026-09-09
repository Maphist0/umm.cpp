#include "umm/session.h"
#include "json.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "umm-cli --model PACKAGE --mode text|image|think-image --prompt TEXT\n"
                             "        [--output image.png] [--max-tokens 256]\n"
                             "        [--width 2048] [--height 2048] [--steps 50] [--cfg 4] [--shift 3] [--seed 42]\n";
                return 0;
            }
            if (key.rfind("--", 0) != 0 || i + 1 == argc || !args.emplace(key, argv[++i]).second) {
                throw std::invalid_argument("Expected unique --option value pairs; use --help");
            }
        }
        const std::vector<std::string> known = {"--model", "--mode", "--prompt",
            "--output", "--max-tokens", "--width", "--height", "--steps", "--cfg", "--shift", "--seed"};
        for (const auto & entry : args) {
            if (std::find(known.begin(), known.end(), entry.first) == known.end()) {
                throw std::invalid_argument("Unknown option: " + entry.first);
            }
        }
        if (!args.count("--model") || !args.count("--prompt")) {
            throw std::invalid_argument("--model and --prompt are required; use --help");
        }
        if (!std::filesystem::is_directory(args.at("--model"))) {
            throw std::invalid_argument("--model must name a model package directory");
        }
        auto get = [&](const std::string & key, const std::string & fallback) {
            return args.count(key) ? args.at(key) : fallback;
        };
        const auto mode = get("--mode", "text");
        if (mode != "text" && mode != "image" && mode != "think-image") {
            throw std::invalid_argument("Mode must be text, image, or think-image");
        }
        umm::session session(args.at("--model"));
        if (mode == "text") {
            std::cout << session.text(args.at("--prompt"), std::stoi(get("--max-tokens", "256"))) << '\n';
            return 0;
        }
        umm::image_options options;
        options.width = std::stoi(get("--width", "2048"));
        options.height = std::stoi(get("--height", "2048"));
        options.steps = std::stoi(get("--steps", "50"));
        options.guidance = std::stof(get("--cfg", "4"));
        options.flow_shift = std::stof(get("--shift", "3"));
        options.seed = std::stoll(get("--seed", "42"));
        options.think = mode == "think-image";
        options.max_think_tokens = std::stoi(get("--max-tokens", "1024"));
        auto image = session.image(args.at("--prompt"), options);
        const std::filesystem::path output = get("--output", "image.png");
        if (!stbi_write_png(output.string().c_str(), image.width, image.height, 3, image.rgb.data(), image.width*3)) {
            throw std::runtime_error("Could not write image");
        }
        nlohmann::json metadata = {{"mode", mode}, {"prompt", args.at("--prompt")},
            {"width", options.width}, {"height", options.height}, {"steps", options.steps},
            {"guidance", options.guidance}, {"flow_shift", options.flow_shift}, {"seed", options.seed},
            {"reasoning", image.reasoning}, {"reasoning_tokens", image.reasoning_tokens},
            {"prefix_tokens", image.prefix_tokens}};
        std::ofstream(output.string() + ".json") << metadata.dump(2) << '\n';
        std::cout << image.reasoning << '\n' << output << '\n';
    } catch (const std::exception & e) {
        std::cerr << "umm: " << e.what() << '\n';
        return 1;
    }
}
