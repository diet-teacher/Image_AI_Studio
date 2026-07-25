#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace phase0 {

// Shared --flag value parsing for both runners. Backend-agnostic: knows
// nothing about TorchScript or AOTInductor.
struct RunnerArgs {
    std::string model;
    std::string input_bin;
    std::string input_meta;
    std::string output_bin;
    std::string output_meta;
    std::string device = "cpu";
    int warmup = 10;
    int repeat = 100;
};

class CommandLineError : public std::runtime_error {
public:
    explicit CommandLineError(const std::string& message) : std::runtime_error(message) {}
};

// Throws CommandLineError with a precise message if a required flag is
// missing -- callers must not swallow it silently.
RunnerArgs parse_runner_args(int argc, char** argv);

}  // namespace phase0
