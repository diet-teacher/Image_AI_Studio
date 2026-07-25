#include "TensorFileIO.h"

#include <fstream>
#include <regex>
#include <sstream>

namespace phase0 {

namespace {

std::string read_whole_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw TensorIOError("Could not open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Minimal, schema-specific JSON field extraction. Not a general parser --
// input.json / *_meta.json always follow the fixed flat shape written by
// image_ai_studio.parity.tensor_io.save_tensor, so this is deliberately
// narrow rather than pulling in a JSON dependency.
std::string extract_string_field(const std::string& text, const std::string& key) {
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, re)) {
        throw TensorIOError("JSON field '" + key + "' not found");
    }
    return match[1].str();
}

int64_t extract_int_field(const std::string& text, const std::string& key) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, re)) {
        throw TensorIOError("JSON field '" + key + "' not found");
    }
    return std::stoll(match[1].str());
}

bool extract_bool_field(const std::string& text, const std::string& key) {
    std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(text, match, re)) {
        throw TensorIOError("JSON field '" + key + "' not found");
    }
    return match[1].str() == "true";
}

std::vector<int64_t> extract_shape_field(const std::string& text) {
    std::regex re("\"shape\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, re)) {
        throw TensorIOError("JSON field 'shape' not found");
    }
    std::vector<int64_t> shape;
    std::stringstream ss(match[1].str());
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            shape.push_back(std::stoll(item));
        }
    }
    return shape;
}

}  // namespace

int64_t element_count_from_shape(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (auto dim : shape) {
        count *= dim;
    }
    return count;
}

TensorMeta read_tensor_meta(const std::string& json_path) {
    std::string text = read_whole_file(json_path);

    TensorMeta meta;
    meta.shape = extract_shape_field(text);
    meta.dtype = extract_string_field(text, "dtype");
    meta.layout = extract_string_field(text, "layout");
    meta.byte_order = extract_string_field(text, "byte_order");
    meta.contiguous = extract_bool_field(text, "contiguous");
    meta.element_count = extract_int_field(text, "element_count");

    if (meta.dtype != "float32") {
        throw TensorIOError("Unsupported dtype '" + meta.dtype +
                             "': Phase 0 only supports float32, got file " + json_path);
    }
    if (meta.element_count != element_count_from_shape(meta.shape)) {
        throw TensorIOError("element_count in " + json_path +
                             " does not match product of shape dims");
    }
    return meta;
}

std::vector<float> read_tensor_bin(const std::string& bin_path, const TensorMeta& meta) {
    std::ifstream in(bin_path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw TensorIOError("Could not open binary file: " + bin_path);
    }
    std::streamsize size_bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    int64_t expected_bytes = meta.element_count * static_cast<int64_t>(sizeof(float));
    if (size_bytes != expected_bytes) {
        throw TensorIOError(bin_path + ": size mismatch, file has " +
                             std::to_string(size_bytes) + " bytes, element_count*4 = " +
                             std::to_string(expected_bytes) + " bytes");
    }

    std::vector<float> data(static_cast<size_t>(meta.element_count));
    if (!in.read(reinterpret_cast<char*>(data.data()), size_bytes)) {
        throw TensorIOError("Failed to read binary payload from: " + bin_path);
    }
    return data;
}

void write_tensor(const std::string& bin_path,
                   const std::string& json_path,
                   const std::vector<float>& data,
                   const std::vector<int64_t>& shape,
                   const std::string& layout) {
    int64_t expected = element_count_from_shape(shape);
    if (static_cast<int64_t>(data.size()) != expected) {
        throw TensorIOError("write_tensor: data size does not match shape");
    }

    std::ofstream bin_out(bin_path, std::ios::binary);
    if (!bin_out) {
        throw TensorIOError("Could not open for write: " + bin_path);
    }
    bin_out.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(float)));

    std::ofstream json_out(json_path);
    if (!json_out) {
        throw TensorIOError("Could not open for write: " + json_path);
    }
    json_out << "{\n";
    json_out << "  \"shape\": [";
    for (size_t i = 0; i < shape.size(); ++i) {
        json_out << shape[i];
        if (i + 1 < shape.size()) json_out << ", ";
    }
    json_out << "],\n";
    json_out << "  \"dtype\": \"float32\",\n";
    json_out << "  \"layout\": \"" << layout << "\",\n";
    json_out << "  \"byte_order\": \"little_endian\",\n";
    json_out << "  \"contiguous\": true,\n";
    json_out << "  \"element_count\": " << expected << "\n";
    json_out << "}\n";
}

}  // namespace phase0
