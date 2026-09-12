// Sampling adapted from karpathy/llama2.c. See THIRD_PARTY.md.
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "kernels.h"

namespace unremarkable {

Sampler::Sampler(int vocabulary, float temperature, float top_p, uint64_t seed)
    : temperature_(temperature), top_p_(top_p), rng_state_(seed) {
  if (vocabulary < 2 || !std::isfinite(temperature) || temperature < 0 || !std::isfinite(top_p) ||
      top_p < 0 || top_p > 1 || seed == 0) {
    throw std::runtime_error("invalid sampling options");
  }
  candidates_.resize(vocabulary);
}

float Sampler::random_float() {
  // xorshift64*, matching the reference's random stream.
  rng_state_ ^= rng_state_ >> 12;
  rng_state_ ^= rng_state_ << 25;
  rng_state_ ^= rng_state_ >> 27;
  uint32_t value = (rng_state_ * 0x2545F4914F6CDD1DULL) >> 32;
  return (value >> 8) / 16777216.0f;
}

int Sampler::sample(std::span<float> logits) {
  if (logits.size() != candidates_.size()) {
    throw std::runtime_error("sampler vocabulary size mismatch");
  }
  for (float value : logits) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite model logits");
    }
  }
  if (temperature_ == 0) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  for (float& value : logits) {
    value /= temperature_;
    if (!std::isfinite(value)) {
      throw std::runtime_error("temperature scaling overflowed");
    }
  }
  softmax_inplace(logits.data(), static_cast<int>(logits.size()));
  const float coin = random_float();
  if (top_p_ == 0 || top_p_ == 1) {
    float cumulative = 0;
    for (size_t i = 0; i < logits.size(); i++) {
      cumulative += logits[i];
      if (coin < cumulative) {
        return static_cast<int>(i);
      }
    }
    return static_cast<int>(logits.size() - 1);
  }

  size_t count = 0;
  const float cutoff = (1.0f - top_p_) / (logits.size() - 1);
  for (size_t i = 0; i < logits.size(); i++) {
    if (logits[i] >= cutoff) {
      candidates_[count++] = {logits[i], static_cast<int>(i)};
    }
  }
  if (!count) {
    throw std::runtime_error("empty sampling distribution");
  }
  std::sort(
      candidates_.begin(), candidates_.begin() + count, [](const Candidate& a, const Candidate& b) {
        return a.probability != b.probability ? a.probability > b.probability : a.token < b.token;
      });
  float cumulative = 0;
  size_t kept = 0;
  do {
    cumulative += candidates_[kept++].probability;
  } while (kept < count && cumulative <= top_p_);
  const float target = coin * cumulative;
  cumulative = 0;
  for (size_t i = 0; i < kept; i++) {
    cumulative += candidates_[i].probability;
    if (target < cumulative) {
      return candidates_[i].token;
    }
  }
  return candidates_[kept - 1].token;
}

}  // namespace unremarkable
