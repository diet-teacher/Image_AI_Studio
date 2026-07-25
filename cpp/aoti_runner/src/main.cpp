// Independent AOTInductor (torch.export + .pt2 package) C++ runner.
// Built and used only because cpp/aoti_probe reported SUPPORTED on this
// machine. No TorchScript fallback exists here -- a failure must be
// reported as FAIL/UNSUPPORTED, never silently routed through
// run_torchscript.
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "CommandLine.h"
#include "ResultWriter.h"
#include "TensorFileIO.h"
#include "TimingStats.h"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void sync_if_cuda(const std::string& device) {
    if (device == "cuda") {
        torch::cuda::synchronize();
    }
}

}  // namespace

int main(int argc, char** argv) {
    phase0::RunnerArgs args;
    try {
        args = phase0::parse_runner_args(argc, argv);
    } catch (const phase0::CommandLineError& e) {
        std::cerr << "[run_aoti] argument error: " << e.what() << "\n";
        return 1;
    }

    phase0::RunResult result;
    result.runner = "aoti";
    result.model_path = args.model;
    result.device = args.device;

    if (args.device == "cuda" && !torch::cuda::is_available()) {
        std::cerr << "[run_aoti] CUDA requested but torch::cuda::is_available() == false. "
                      "Not falling back to CPU.\n";
        result.status = "UNSUPPORTED";
        result.error_message = "CUDA requested but not available on this build/machine";
        phase0::write_run_result(args.output_meta + ".run_result.json", result);
        return 1;
    }

    try {
        c10::DeviceIndex device_index = args.device == "cuda" ? 0 : -1;

        auto load_start = Clock::now();
        torch::inductor::AOTIModelPackageLoader loader(args.model, "model", false, 1, device_index);
        sync_if_cuda(args.device);
        auto load_end = Clock::now();
        result.model_load_ms = elapsed_ms(load_start, load_end);

        phase0::TensorMeta input_meta = phase0::read_tensor_meta(args.input_meta);
        std::vector<float> input_data = phase0::read_tensor_bin(args.input_bin, input_meta);
        std::vector<int64_t> shape(input_meta.shape.begin(), input_meta.shape.end());

        torch::Device device(args.device == "cuda" ? torch::kCUDA : torch::kCPU);
        at::Tensor input_cpu = torch::from_blob(input_data.data(), shape, torch::kFloat32).clone();
        at::Tensor input = input_cpu.to(device);

        // First inference, timed separately from warmup/measurement.
        auto first_start = Clock::now();
        std::vector<at::Tensor> outputs = loader.run({input});
        sync_if_cuda(args.device);
        auto first_end = Clock::now();
        result.first_inference_ms = elapsed_ms(first_start, first_end);

        if (outputs.empty()) {
            throw std::runtime_error("AOTIModelPackageLoader::run() returned no outputs");
        }

        phase0::TimingStats warmup_stats;
        for (int i = 0; i < args.warmup; ++i) {
            auto s = Clock::now();
            outputs = loader.run({input});
            sync_if_cuda(args.device);
            auto e = Clock::now();
            warmup_stats.add_sample_ms(elapsed_ms(s, e));
        }
        result.warmup_timing = warmup_stats.summarize();

        phase0::TimingStats measured_stats;
        std::vector<int64_t> out_shape;
        at::Tensor out_cpu;
        for (int i = 0; i < args.repeat; ++i) {
            auto s = Clock::now();
            outputs = loader.run({input});
            sync_if_cuda(args.device);
            auto e = Clock::now();
            measured_stats.add_sample_ms(elapsed_ms(s, e));

            out_cpu = outputs[0].to(torch::kCPU).contiguous().to(torch::kFloat32);
            if (i == 0) {
                out_shape.assign(out_cpu.sizes().begin(), out_cpu.sizes().end());
            } else {
                std::vector<int64_t> this_shape(out_cpu.sizes().begin(), out_cpu.sizes().end());
                if (this_shape != out_shape) {
                    throw std::runtime_error("output shape changed across repeated inference calls");
                }
            }
        }
        result.measured_timing = measured_stats.summarize();

        std::vector<float> out_data(
            out_cpu.data_ptr<float>(), out_cpu.data_ptr<float>() + out_cpu.numel());
        phase0::write_tensor(args.output_bin, args.output_meta, out_data, out_shape, "NC");

        result.status = "PASS";
        std::cout << "[run_aoti] PASS: model_load_ms=" << result.model_load_ms
                   << " first_inference_ms=" << result.first_inference_ms
                   << " mean_measured_ms=" << result.measured_timing.mean_ms << "\n";
    } catch (const std::exception& e) {
        result.status = "FAIL";
        result.error_message = e.what();
        std::cerr << "[run_aoti] FAIL: " << e.what() << "\n";
        std::cerr << "  runner=aoti model=" << args.model << " device=" << args.device
                   << " input=" << args.input_bin << "\n";
        phase0::write_run_result(args.output_meta + ".run_result.json", result);
        return 1;
    }

    phase0::write_run_result(args.output_meta + ".run_result.json", result);
    return 0;
}
