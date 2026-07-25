#pragma once

#include <optional>
#include <string>

#include "TimingStats.h"

namespace phase0 {

// One inference run's outcome, backend-agnostic. Runners fill this in
// and ResultWriter serializes it -- no backend-specific fields here.
struct RunResult {
    std::string runner;   // "torchscript" | "aoti"
    std::string model_path;
    std::string device;
    std::string status;   // PASS | FAIL | UNSUPPORTED | INVALID_BUILD_CONFIGURATION
    std::optional<std::string> error_message;

    double model_load_ms = 0.0;
    double first_inference_ms = 0.0;
    TimingSummary warmup_timing;
    TimingSummary measured_timing;
};

void write_run_result(const std::string& json_path, const RunResult& result);

}  // namespace phase0
