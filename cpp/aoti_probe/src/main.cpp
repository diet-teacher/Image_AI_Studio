// Minimal AOTInductor C++ runtime capability probe.
//
// If this file fails to compile at all, the header itself is missing
// or incompatible -> HEADER_NOT_FOUND / COMPILE_FAILED (see build log).
// If it compiles but fails to link -> LIBRARY_OR_SYMBOL_NOT_FOUND /
// LINK_FAILED. If it builds and runs but the package fails to load or
// run -> PACKAGE_LOAD_FAILED. Otherwise -> SUPPORTED.
//
// This target intentionally does not share any code path with
// cpp/torchscript_runner.
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "TensorFileIO.h"

namespace {

std::unordered_map<std::string, std::string> parse_flags(int argc, char** argv) {
    std::unordered_map<std::string, std::string> flags;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0 && i + 1 < argc) {
            flags[arg.substr(2)] = argv[++i];
        }
    }
    return flags;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "[aoti_probe] header include: OK (compiled)\n";

    auto flags = parse_flags(argc, argv);
    auto package_it = flags.find("package");
    auto input_bin_it = flags.find("input-bin");
    auto input_meta_it = flags.find("input-meta");

    if (package_it == flags.end() || input_bin_it == flags.end() ||
        input_meta_it == flags.end()) {
        std::cerr << "Usage: probe_aoti --package <model.pt2> --input-bin <input.bin> "
                     "--input-meta <input.json>\n";
        std::cerr << "[aoti_probe] result: COMPILE_LINK_ONLY (no package path given, "
                     "skipping package load stage)\n";
        return 2;
    }

    try {
        // Constructing this object forces the linker to resolve
        // torch::inductor::AOTIModelPackageLoader symbols -> proves LINK, not
        // just COMPILE.
        torch::inductor::AOTIModelPackageLoader loader(package_it->second);
        std::cout << "[aoti_probe] AOTIModelPackageLoader construction: OK (link resolved)\n";

        phase0::TensorMeta meta = phase0::read_tensor_meta(input_meta_it->second);
        std::vector<float> input_data = phase0::read_tensor_bin(input_bin_it->second, meta);

        std::vector<int64_t> shape(meta.shape.begin(), meta.shape.end());
        at::Tensor input = torch::from_blob(input_data.data(), shape, torch::kFloat32).clone();

        std::vector<at::Tensor> outputs = loader.run({input});
        if (outputs.empty()) {
            std::cerr << "[aoti_probe] result: PACKAGE_LOAD_FAILED (run() returned no outputs)\n";
            return 1;
        }

        std::cout << "[aoti_probe] package load + run: OK, output shape = [";
        for (int64_t d : outputs[0].sizes()) std::cout << d << " ";
        std::cout << "]\n";
        std::cout << "[aoti_probe] result: SUPPORTED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[aoti_probe] exception: " << e.what() << "\n";
        std::cerr << "[aoti_probe] result: PACKAGE_LOAD_FAILED\n";
        return 1;
    }
}
