#ifndef UNREMARKABLE_SRC_PROFILE_H_
#define UNREMARKABLE_SRC_PROFILE_H_

#include <array>
#include <chrono>
#include <cstdio>
#include <string>

namespace unremarkable {

// Timing categories shared by forward() and prefill().
enum class Stage {
  kEmbedding,
  kNorm,
  kQkv,
  kRope,
  kAttention,
  kOutput,
  kResidual,
  kGateUp,
  kSwiGLU,
  kDown,
  kClassifier,
  kCount
};

inline constexpr int kStages = static_cast<int>(Stage::kCount);
inline constexpr std::array<const char*, kStages> kStageNames = {
    "embedding", "norm",    "qkv",    "rope", "attention", "output",
    "residual",  "gate_up", "swiglu", "down", "classifier"};

// Elapsed seconds. Quantize and waiting are included in projection steps;
// worker time overlaps the caller. See docs/profile.md for counter boundaries.
struct Profile {
  std::array<double, kStages> steps{};
  double total = 0;
  double quantize = 0;
  double waiting = 0;
  double worker = 0;
  int tokens = 0;

  Profile operator-(const Profile& earlier) const {
    Profile difference = *this;
    for (int i = 0; i < kStages; ++i) {
      difference.steps[i] -= earlier.steps[i];
    }
    difference.total -= earlier.total;
    difference.quantize -= earlier.quantize;
    difference.waiting -= earlier.waiting;
    difference.worker -= earlier.worker;
    difference.tokens -= earlier.tokens;
    return difference;
  }

  std::string json() const {
    auto ms = [](double seconds) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.3f", seconds * 1000);
      return std::string(buffer);
    };
    double attributed = 0;
    std::string out =
        "{\"tokens\":" + std::to_string(tokens) + ",\"total_ms\":" + ms(total) + ",\"steps_ms\":{";
    for (int i = 0; i < kStages; ++i) {
      attributed += steps[i];
      out += (i ? ",\"" : "\"") + std::string(kStageNames[i]) + "\":" + ms(steps[i]);
    }
    return out + "},\"unattributed_ms\":" + ms(total - attributed) +
           ",\"quantize_ms\":" + ms(quantize) + ",\"waiting_ms\":" + ms(waiting) +
           ",\"worker_ms\":" + ms(worker) + "}";
  }
};

#ifdef UNREMARKABLE_PROFILE
// Each lap records elapsed time since the previous marker.
class Profiler {
 public:
  void begin() { start_ = last_ = now(); }
  void lap(Stage stage) {
    const double time = now();
    totals_.steps[static_cast<int>(stage)] += time - last_;
    last_ = time;
  }
  double clock() const { return now(); }
  void quantized(double started) { totals_.quantize += now() - started; }
  void end(int tokens = 1) {
    totals_.total += now() - start_;
    totals_.tokens += tokens;
  }
  const Profile& totals() const { return totals_; }

 private:
  static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  Profile totals_;
  double start_ = 0;
  double last_ = 0;
};
#else
// Timers are disabled in normal builds.
class Profiler {
 public:
  void begin() {}
  void lap(Stage) {}
  double clock() const { return 0; }
  void quantized(double) {}
  void end(int = 1) {}
};
#endif

}  // namespace unremarkable
#endif
