#include "CommandLine.h"

#include <unordered_map>

namespace phase0 {

namespace {

std::unordered_map<std::string, std::string> parse_flags(int argc, char** argv) {
    std::unordered_map<std::string, std::string> flags;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            throw CommandLineError("Unexpected positional argument: " + arg);
        }
        std::string key = arg.substr(2);
        if (i + 1 >= argc) {
            throw CommandLineError("Flag --" + key + " is missing a value");
        }
        flags[key] = argv[++i];
    }
    return flags;
}

std::string require(const std::unordered_map<std::string, std::string>& flags,
                     const std::string& key) {
    auto it = flags.find(key);
    if (it == flags.end()) {
        throw CommandLineError("Missing required flag: --" + key);
    }
    return it->second;
}

}  // namespace

RunnerArgs parse_runner_args(int argc, char** argv) {
    auto flags = parse_flags(argc, argv);

    RunnerArgs args;
    args.model = require(flags, "model");
    args.input_bin = require(flags, "input-bin");
    args.input_meta = require(flags, "input-meta");
    args.output_bin = require(flags, "output-bin");
    args.output_meta = require(flags, "output-meta");

    if (auto it = flags.find("device"); it != flags.end()) {
        args.device = it->second;
    }
    if (args.device != "cpu" && args.device != "cuda") {
        throw CommandLineError("--device must be 'cpu' or 'cuda', got '" + args.device + "'");
    }

    if (auto it = flags.find("warmup"); it != flags.end()) {
        args.warmup = std::stoi(it->second);
    }
    if (auto it = flags.find("repeat"); it != flags.end()) {
        args.repeat = std::stoi(it->second);
    }

    return args;
}

}  // namespace phase0
