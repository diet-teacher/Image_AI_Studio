#include "ResultWriter.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace phase0 {

namespace {

std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

void write_timing(std::ofstream& out, const char* name, const TimingSummary& t) {
    out << "  \"" << name << "\": {\n"
        << "    \"count\": " << t.count << ",\n"
        << "    \"min_ms\": " << t.min_ms << ",\n"
        << "    \"max_ms\": " << t.max_ms << ",\n"
        << "    \"mean_ms\": " << t.mean_ms << ",\n"
        << "    \"stddev_ms\": " << t.stddev_ms << "\n"
        << "  }";
}

}  // namespace

void write_run_result(const std::string& json_path, const RunResult& result) {
    std::filesystem::path path(json_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(json_path);
    if (!out) {
        throw std::runtime_error("Could not open for write: " + json_path);
    }

    out << "{\n";
    out << "  \"runner\": \"" << escape_json(result.runner) << "\",\n";
    out << "  \"model_path\": \"" << escape_json(result.model_path) << "\",\n";
    out << "  \"device\": \"" << escape_json(result.device) << "\",\n";
    out << "  \"status\": \"" << escape_json(result.status) << "\",\n";
    out << "  \"error_message\": "
        << (result.error_message ? ("\"" + escape_json(*result.error_message) + "\"") : "null")
        << ",\n";
    out << "  \"model_load_ms\": " << result.model_load_ms << ",\n";
    out << "  \"first_inference_ms\": " << result.first_inference_ms << ",\n";
    write_timing(out, "warmup_timing", result.warmup_timing);
    out << ",\n";
    write_timing(out, "measured_timing", result.measured_timing);
    out << "\n}\n";
}

}  // namespace phase0
