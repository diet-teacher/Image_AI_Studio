#include "TimingStats.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace phase0 {

void TimingStats::add_sample_ms(double duration_ms) {
    samples_.push_back(duration_ms);
}

TimingSummary TimingStats::summarize() const {
    TimingSummary summary;
    summary.count = static_cast<int>(samples_.size());
    if (samples_.empty()) {
        return summary;
    }

    summary.min_ms = *std::min_element(samples_.begin(), samples_.end());
    summary.max_ms = *std::max_element(samples_.begin(), samples_.end());

    double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
    summary.mean_ms = sum / static_cast<double>(samples_.size());

    double sq_sum = 0.0;
    for (double v : samples_) {
        double diff = v - summary.mean_ms;
        sq_sum += diff * diff;
    }
    summary.stddev_ms = samples_.size() > 1
                             ? std::sqrt(sq_sum / static_cast<double>(samples_.size() - 1))
                             : 0.0;

    return summary;
}

}  // namespace phase0
