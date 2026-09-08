#pragma once

#include <string>

namespace umm {
std::string text_prompt(const std::string & user);
std::string image_prompt(const std::string & user, bool think);
std::string unconditional_prompt();
}
