#include "prompts.h"

namespace umm {

static const std::string generation_system =
    "You are an image generation and editing assistant that accurately understands and executes user intent.\n\n"
    "You support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a "
    "<think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. "
    "Keep it reasonable and directly useful for producing the final image.\n\n2. Non-Think Mode:\nIf no reasoning "
    "is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a "
    "high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and "
    "completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\nB. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an "
    "edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly "
    "changed.\n\nGeneral Rules:\n- For any visible text in the image, follow the language specified for the rendered "
    "text in the user's description, not the language of the prompt. If no language is specified, use the user's input "
    "language.";

static std::string chat_prompt(const std::string & user, const std::string & system) {
    std::string result;
    if (!system.empty()) {
        result = "<|im_start|>system\n" + system + "<|im_end|>\n";
    }
    return result + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string text_prompt(const std::string & user) {
    return chat_prompt(user, "");
}

std::string image_prompt(const std::string & user, bool think) {
    return chat_prompt(user, generation_system) + (think ? "<think>\n" : "<think>\n\n</think>\n\n<img>");
}

std::string unconditional_prompt() {
    return chat_prompt("", "") + "<img>";
}

}
