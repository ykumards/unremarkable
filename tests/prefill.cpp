#include <algorithm>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine.h"

using unremarkable::Engine;

void equal(std::span<const float> expected, std::span<const float> actual) {
  if (expected.size() != actual.size() ||
      !std::equal(expected.begin(), expected.end(), actual.begin())) {
    throw std::runtime_error("batched prefill changed logits or cached history");
  }
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: test-prefill MODEL");
    }
    const std::vector<int> tokens{1, 19, 43, 7, 4, 9, 15, 27, 39, 5, 12, 6, 21, 3, 8, 18, 2};
    for (int threads : {1, 2}) {
      Engine reference(argv[1], 18, threads);
      Engine batched(argv[1], 18, threads);
      for (int batch : {1, 2, 3, 4, 8}) {
        for (int prefix : {0, 1, 3}) {
          for (int count : {1, 7, 8, 9, 14}) {
            reference.reset();
            batched.reset();
            for (int i = 0; i < prefix; ++i) {
              equal(reference.forward(tokens[i], i), batched.forward(tokens[i], i));
            }
            std::span<const float> expected;
            for (int i = prefix; i < prefix + count; ++i) {
              expected = reference.forward(tokens[i], i);
            }
            equal(expected, batched.prefill(std::span(tokens).subspan(prefix, count), batch));
            // Decode after the batch checks every layer's persisted K/V history.
            equal(reference.forward(11, prefix + count), batched.forward(11, prefix + count));
          }
        }
      }
      // Repeated prefill calls append to the same prefix.
      reference.reset();
      batched.reset();
      for (int i = 0; i < 10; ++i) {
        reference.forward(tokens[i], i);
      }
      batched.prefill(std::span(tokens).first(3), 2);
      batched.prefill(std::span(tokens).subspan(3, 7), 4);
      equal(reference.forward(11, 10), batched.forward(11, 10));

      // Validation must reject the entire request before changing sequence state.
      reference.reset();
      batched.reset();
      reference.forward(1, 0);
      batched.forward(1, 0);
      const std::vector<int> bad_token{2, batched.config().vocab_size};
      const std::vector<int> too_long(18, 1);
      auto reject = [&](std::span<const int> input, int batch) {
        try {
          batched.prefill(input, batch);
        } catch (const std::runtime_error&) {
          return;
        }
        throw std::runtime_error("invalid prefill accepted");
      };
      reject({}, 8);
      reject(bad_token, 8);
      reject(too_long, 8);
      reject(std::span(tokens).first(1), 0);
      reject(std::span(tokens).first(1), 9);
      equal(reference.forward(2, 1), batched.forward(2, 1));
      reference.reset();
      batched.reset();
      std::vector<int> full(18, 1);
      std::span<const float> expected;
      for (int i = 0; i < 18; ++i) {
        expected = reference.forward(1, i);
      }
      equal(expected, batched.prefill(full, 8));
      reject(std::span(tokens).first(1), 1);
    }
    std::puts("prefill: exact logits, prefixes, chunks, decode, reset, and rejection passed");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
