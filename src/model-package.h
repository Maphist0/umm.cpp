#pragma once

#include <string>

namespace umm {

struct model_paths {
    std::string understanding;
    std::string generation;
};

model_paths resolve_model(const std::string & model, const std::string & generation = "");

}
