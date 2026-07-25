#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace phase0 {

// Mirrors image_ai_studio.parity.tensor_io on the Python side: a flat
// float32 binary file + a small JSON sidecar. Phase 0 only supports
// float32 -- any other dtype in the JSON is a hard error, not a silent
// reinterpret.
struct TensorMeta {
    std::vector<int64_t> shape;
    std::string dtype;
    std::string layout;
    std::string byte_order;
    bool contiguous = true;
    int64_t element_count = 0;
};

class TensorIOError : public std::runtime_error {
public:
    explicit TensorIOError(const std::string& message) : std::runtime_error(message) {}
};

TensorMeta read_tensor_meta(const std::string& json_path);

// Reads the raw float32 payload and validates that the file size equals
// element_count * sizeof(float); throws TensorIOError otherwise.
std::vector<float> read_tensor_bin(const std::string& bin_path, const TensorMeta& meta);

void write_tensor(const std::string& bin_path,
                   const std::string& json_path,
                   const std::vector<float>& data,
                   const std::vector<int64_t>& shape,
                   const std::string& layout);

int64_t element_count_from_shape(const std::vector<int64_t>& shape);

}  // namespace phase0
