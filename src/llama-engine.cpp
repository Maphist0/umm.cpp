#include "llama-engine.h"

// Keep the dependency on llama's cache layout in this adapter only.
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace umm {

llama_engine::llama_engine(const std::string & model_path, int context_size, int gpu_layers,
                         bool full_precision) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    overrides_[0].tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
    std::strcpy(overrides_[0].key, "sensenova_u1.full_precision");
    overrides_[0].val_bool = full_precision;
    mp.kv_overrides = overrides_;
    model_.reset(llama_model_load_from_file(model_path.c_str(), mp));
    if (!model_) {
        throw std::runtime_error("Could not load understanding GGUF");
    }
    char architecture[64]{};
    llama_model_meta_val_str(model_.get(), "general.architecture", architecture, sizeof(architecture));
    if (std::string(architecture) != "sensenova_u1") {
        throw std::runtime_error("Expected a SenseNova U1 understanding GGUF");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = context_size;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = full_precision ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    const auto cache_type = llama_model_ftype(model_.get()) == LLAMA_FTYPE_MOSTLY_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
    cp.type_k = cp.type_v = full_precision ? GGML_TYPE_F32 : cache_type;
    context_.reset(llama_init_from_model(model_.get(), cp));
    if (!context_) {
        throw std::runtime_error("Could not create understanding context");
    }
}

std::vector<llama_token> llama_engine::tokenize(const std::string & text) const {
    const auto * vocab = llama_model_get_vocab(model_.get());
    int count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
    if (count >= 0) {
        return {};
    }
    std::vector<llama_token> result(-count);
    count = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), false, true);
    if (count < 0) {
        throw std::runtime_error("Tokenization failed");
    }
    result.resize(count);
    return result;
}

std::string llama_engine::piece(llama_token token) const {
    const auto * vocab = llama_model_get_vocab(model_.get());
    std::string result(32, '\0');
    int size = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    if (size < 0) {
        result.resize(-size);
        size = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    }
    if (size < 0) {
        throw std::runtime_error("Token decoding failed");
    }
    result.resize(size);
    return result;
}

bool llama_engine::is_end(llama_token token) const {
    return llama_vocab_is_eog(llama_model_get_vocab(model_.get()), token);
}

void llama_engine::reset() {
    llama_memory_clear(llama_get_memory(context_.get()), true);
    tokens_.clear();
}

void llama_engine::append(const std::vector<llama_token> & tokens) {
    if (tokens_.size() + tokens.size() > llama_n_ctx(context_.get())) {
        throw std::runtime_error("Understanding context is full");
    }
    for (size_t offset = 0; offset < tokens.size();) {
        const int n = std::min<size_t>(512, tokens.size() - offset);
        std::vector<llama_pos> positions(4*n, 0);
        std::vector<int32_t> seq_counts(n, 1);
        llama_seq_id sequence = 0;
        std::vector<llama_seq_id *> seq_ids(n, &sequence);
        std::vector<int8_t> outputs(n, 0);
        for (int i = 0; i < n; ++i) {
            positions[i] = tokens_.size() + i;
        }
        outputs.back() = 1;
        llama_batch batch{};
        batch.n_tokens = n;
        batch.token = const_cast<llama_token *>(tokens.data() + offset);
        batch.pos = positions.data();
        batch.n_seq_id = seq_counts.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        if (llama_decode(context_.get(), batch) != 0) {
            throw std::runtime_error("Understanding decode failed");
        }
        tokens_.insert(tokens_.end(), tokens.begin() + offset, tokens.begin() + offset + n);
        offset += n;
    }
}

const float * llama_engine::logits() const {
    if (tokens_.empty()) {
        throw std::runtime_error("Prefill the context before reading logits");
    }
    return llama_get_logits_ith(context_.get(), -1);
}

int llama_engine::vocab_size() const {
    return llama_vocab_n_tokens(llama_model_get_vocab(model_.get()));
}

llama_token llama_engine::greedy() const {
    const float * scores = logits();
    return std::max_element(scores, scores + vocab_size()) - scores;
}

prefix_view llama_engine::prefix() const {
    context_->synchronize();
    auto * cache = dynamic_cast<llama_kv_cache *>(context_->get_memory());
    if (!cache || tokens_.empty()) {
        throw std::runtime_error("No exportable attention prefix");
    }
    const auto & cells = cache->get_cells(0);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (!cells.seq_has(i, 0) || cells.pos_get(i) != static_cast<llama_pos>(i)) {
            throw std::runtime_error("Prefix export requires an unshifted, contiguous sequence");
        }
    }
    const auto layers = cache->get_layer_ids();
    prefix_view result;
    const size_t max_nodes = 4*layers.size() + 4;
    result.descriptors.reset(ggml_init({ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true}));
    if (!result.descriptors) {
        throw std::runtime_error("Could not allocate prefix descriptors");
    }
    llama_kv_cache::slot_info slots{};
    slots.s0 = slots.s1 = 0;
    ggml_cgraph * packing = nullptr;
    ggml_backend_dev_t packing_device = nullptr;
    for (uint32_t layer : layers) {
        auto * key = cache->get_k(result.descriptors.get(), layer, tokens_.size(), slots);
        auto * value = cache->get_v(result.descriptors.get(), layer, tokens_.size(), slots);
        ggml_backend_view_init(key);
        ggml_backend_view_init(value);
        if (value->nb[1] > value->nb[2]) {
            value = ggml_permute(result.descriptors.get(), value, 2, 1, 0, 3);
            ggml_backend_view_init(value);
        }
        if (!ggml_are_same_shape(key, value) || !ggml_is_contiguous(key)) {
            throw std::runtime_error("Unsupported prefix cache layout");
        }
        if (!ggml_is_contiguous(value)) {
            auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(value->buffer));
            if (!device && ggml_backend_buffer_is_host(value->buffer)) {
                device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
            if (!device) {
                throw std::runtime_error("Could not identify prefix packing device");
            }
            if (!packing) {
                packing = ggml_new_graph_custom(result.descriptors.get(), max_nodes, false);
                packing_device = device;
            } else if (packing_device != device) {
                throw std::runtime_error("Packing transposed values requires one cache device");
            }
            value = ggml_cont(result.descriptors.get(), value);
            ggml_build_forward_expand(packing, value);
        }
        result.keys.push_back(key);
        result.values.push_back(value);
    }
    if (packing) {
        // Ordinary attention transposes V. Pack it once at the phase boundary.
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
                ggml_backend_dev_init(packing_device, nullptr), ggml_backend_free);
        if (!backend) {
            throw std::runtime_error("Could not initialize prefix packing backend");
        }
        result.storage.reset(ggml_backend_alloc_ctx_tensors(result.descriptors.get(), backend.get()));
        if (!result.storage || ggml_backend_graph_compute(backend.get(), packing) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Could not pack prefix values");
        }
        ggml_backend_synchronize(backend.get());
    }
    return result;
}

}
