#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

namespace umm {

// Views remain valid until the next decode/reset or destruction of the engine.
struct prefix_view {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> descriptors{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage{nullptr, ggml_backend_buffer_free};
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
};

class llama_engine {
public:
    explicit llama_engine(const std::string & model_path, int context_size = 4096, int gpu_layers = 99,
                          bool full_precision = false);
    std::vector<llama_token> tokenize(const std::string & text) const;
    std::string piece(llama_token token) const;
    bool is_end(llama_token token) const;
    void reset();
    void append(const std::vector<llama_token> & tokens);
    llama_token greedy() const;
    const float * logits() const;
    int vocab_size() const;
    prefix_view prefix() const;
    const std::vector<llama_token> & tokens() const { return tokens_; }

private:
    llama_model_kv_override overrides_[2]{};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, llama_free};
    std::vector<llama_token> tokens_;
};

}
