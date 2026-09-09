#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace umm {

struct image_options {
    int width = 2048;
    int height = 2048;
    int steps = 50;
    float guidance = 4.0f;
    float flow_shift = 3.0f;
    int64_t seed = 42;
    bool think = false;
    int max_think_tokens = 1024;
};

struct image_result {
    int width;
    int height;
    std::vector<uint8_t> rgb;
    std::string reasoning;
    std::vector<int32_t> reasoning_tokens;
    std::vector<int32_t> prefix_tokens;
};

class session {
public:
    // Accept a model package directory, or an understanding GGUF and optional generation checkpoint.
    session(const std::string & model, const std::string & generation_model = "");
    ~session();
    session(const session &) = delete;
    session & operator=(const session &) = delete;

    std::string text(const std::string & prompt, int max_tokens = 256);
    image_result image(const std::string & prompt, const image_options & options = {});

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}
