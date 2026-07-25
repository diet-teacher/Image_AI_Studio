#pragma once

#include <vector>

namespace phase0 {

struct TimingSummary {
    int count = 0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
};

// Accumulates per-iteration durations (milliseconds) measured by the
// caller with std::chrono::steady_clock (CPU) or CUDA event/synchronize
// (GPU) -- this class does no timing itself, only aggregation, so it
// works identically for both device paths.
class TimingStats {
public:
    void add_sample_ms(double duration_ms);
    TimingSummary summarize() const;
    const std::vector<double>& samples() const { return samples_; }

private:
    std::vector<double> samples_;
};

}  // namespace phase0
