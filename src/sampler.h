#ifndef UNREMARKABLE_SRC_SAMPLER_H_
#define UNREMARKABLE_SRC_SAMPLER_H_

#include <cstdint>
#include <span>
#include <vector>

namespace unremarkable {

class Sampler {
 public:
  Sampler(int vocabulary, float temperature = 0, float top_p = 0.9f, uint64_t seed = 1);
  // Temperature sampling replaces logits with probabilities in place.
  int sample(std::span<float> logits);

 private:
  struct Candidate {
    float probability;
    int token;
  };
  float random_float();
  std::vector<Candidate> candidates_;
  float temperature_;
  float top_p_;
  uint64_t rng_state_;
};

}  // namespace unremarkable
#endif
