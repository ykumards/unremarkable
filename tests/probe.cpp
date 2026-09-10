// Dumps logits for comparison with an independent Python forward pass.
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      throw std::runtime_error("usage: probe MODEL TOKEN... | probe tokenize TOK VOCAB TEXT");
    }
    // Byte-level tokenizer check: emit the identifiers for one line of text.
    if (std::string(argv[1]) == "tokenize") {
      if (argc != 5) {
        throw std::runtime_error("usage: probe tokenize TOKENIZER VOCAB TEXT");
      }
      unremarkable::Tokenizer tokenizer(argv[2], std::stoi(argv[3]));
      for (int id : tokenizer.encode(argv[4])) {
        std::cout << id << ' ';
      }
      std::cout << '\n';
      return 0;
    }
    // UNREMARKABLE_THREADS lets the tests run the same fixture on several
    // threads and compare: the logits must not depend on how many there are.
    const char* threads = std::getenv("UNREMARKABLE_THREADS");
    unremarkable::Engine engine(argv[1], 0, threads ? std::stoi(threads) : 1);
    std::vector<float> first;
    std::cout << std::setprecision(9);
    for (int i = 2; i < argc; i++) {
      auto logits = engine.forward(std::stoi(argv[i]), i - 2);
      if (i == 2) {
        first.assign(logits.begin(), logits.end());
      }
      for (float value : logits) {
        std::cout << value << ' ';
      }
      std::cout << '\n';
    }
    engine.reset();
    auto reset = engine.forward(std::stoi(argv[2]), 0);
    if (!std::equal(first.begin(), first.end(), reset.begin())) {
      throw std::runtime_error("reset changed first-position logits");
    }
    bool rejected = false;
    try {
      engine.forward(1, 3);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error("out-of-order forward was accepted");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
