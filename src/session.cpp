#include "umm/session.h"
#include "llama-engine.h"
#include "model-package.h"
#include "prompts.h"
#include "stable-diffusion.h"
#include "ggml-backend.h"

#include <stdexcept>

namespace umm {

struct session::impl {
    llama_engine text_engine;
    std::string generation_model;
    std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> image_engine{nullptr, free_sd_ctx};

    impl(const std::string & text_model, const std::string & image_model)
        : text_engine(text_model, 4096, 99), generation_model(image_model) {}

    void load_image_engine() {
        if (image_engine) {
            return;
        }
        if (generation_model.empty()) {
            throw std::runtime_error("Provide the generation checkpoint for image output");
        }
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        params.model_path = generation_model.c_str();
        params.n_threads = 8;
        params.enable_mmap = true;
        params.flash_attn = params.diffusion_flash_attn = true;
        params.external_kv_prefix = true;
        image_engine.reset(new_sd_ctx(&params));
        if (!image_engine) {
            throw std::runtime_error("Could not load generation checkpoint");
        }
    }

    void transfer_prefix(bool unconditional) {
        auto prefix = text_engine.prefix();
        const auto & tokens = text_engine.tokens();
        const sd_kv_prefix_t data{tokens.data(), tokens.size(), prefix.keys.data(), prefix.values.data(), prefix.keys.size()};
        if (!sd_set_kv_prefix(image_engine.get(), unconditional, &data)) {
            throw std::runtime_error("Could not transfer understanding prefix to generation");
        }
    }
};

session::session(const std::string & model, const std::string & generation_model) {
    const auto paths = resolve_model(model, generation_model);
    ggml_backend_load_all();
    llama_backend_init();
    impl_ = std::make_unique<impl>(paths.understanding, paths.generation);
}

session::~session() = default;

std::string session::text(const std::string & prompt, int max_tokens) {
    if (max_tokens < 1) {
        throw std::invalid_argument("max_tokens must be positive");
    }
    auto & engine = impl_->text_engine;
    engine.reset();
    engine.append(engine.tokenize(text_prompt(prompt)));
    std::string result;
    for (int i = 0; i < max_tokens; ++i) {
        const auto token = engine.greedy();
        if (engine.is_end(token)) {
            break;
        }
        if (engine.piece(token) == "<img>") {
            throw std::runtime_error("The model requested image output during a text-only request");
        }
        result += engine.piece(token);
        engine.append({token});
    }
    return result;
}

image_result session::image(const std::string & prompt, const image_options & options) {
    if (options.width < 32 || options.height < 32 || options.width % 32 || options.height % 32 ||
        options.steps < 1 || options.guidance < 1 || options.flow_shift <= 0 || options.max_think_tokens < 1) {
        throw std::invalid_argument("Use dimensions divisible by 32, positive steps/shift, and guidance >= 1");
    }
    impl_->load_image_engine();
    auto & engine = impl_->text_engine;
    if (options.guidance > 1) {
        engine.reset();
        engine.append(engine.tokenize(unconditional_prompt()));
        impl_->transfer_prefix(true);
    }
    engine.reset();
    engine.append(engine.tokenize(image_prompt(prompt, options.think)));
    image_result result{options.width, options.height, {}, {}, {}, {}};
    if (options.think) {
        // Match the official t2i_generate/_generate_think boundary, including its token cap.
        for (int i = 0; i < options.max_think_tokens; ++i) {
            const auto token = engine.greedy();
            const auto piece = engine.piece(token);
            if (piece == "<|im_end|>") {
                break;
            }
            result.reasoning += piece;
            result.reasoning_tokens.push_back(token);
            engine.append({token});
            if (piece == "</think>") {
                break;
            }
        }
        engine.append(engine.tokenize("\n\n<img>"));
    }
    result.prefix_tokens = engine.tokens();
    impl_->transfer_prefix(false);

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt.c_str();
    params.negative_prompt = "";
    params.width = options.width;
    params.height = options.height;
    params.seed = options.seed;
    params.batch_count = 1;
    params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    params.sample_params.scheduler = sd_get_default_scheduler(impl_->image_engine.get(), EULER_SAMPLE_METHOD);
    params.sample_params.sample_steps = options.steps;
    params.sample_params.guidance.txt_cfg = options.guidance;
    params.sample_params.flow_shift = options.flow_shift;

    sd_image_t * images = nullptr;
    int count = 0;
    const bool success = generate_image(impl_->image_engine.get(), &params, &images, &count);
    if (!success || count != 1 || !images || images[0].channel != 3) {
        free_sd_images(images, count);
        throw std::runtime_error("Image generation failed");
    }
    result.rgb.assign(images[0].data, images[0].data + size_t(result.width)*result.height*3);
    free_sd_images(images, count);
    return result;
}

}
