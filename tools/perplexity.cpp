// Compares a candidate checkpoint's next-token predictions with a reference's on
// the same text, in the style of llama.cpp's perplexity --kl-divergence.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "tokenizer.h"

namespace {

constexpr int kContext = 512;

// Log-probabilities in double precision; returns the most likely token.
int log_softmax(std::span<const float> logits, std::vector<double>& output) {
  const auto top = std::max_element(logits.begin(), logits.end());
  const double maximum = *top;
  double sum = 0;
  for (float logit : logits) {
    sum += std::exp(logit - maximum);
  }
  const double log_sum = maximum + std::log(sum);
  output.resize(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    output[i] = logits[i] - log_sum;
  }
  return static_cast<int>(top - logits.begin());
}

struct Estimate {
  double mean;
  double standard_error;
};

Estimate estimate(const std::vector<double>& values) {
  double mean = 0;
  for (double value : values) {
    mean += value;
  }
  mean /= values.size();
  double squares = 0;
  for (double value : values) {
    squares += (value - mean) * (value - mean);
  }
  const double variance = values.size() > 1 ? squares / (values.size() - 1) : 0;
  return {mean, std::sqrt(variance / values.size())};
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(fraction * (values.size() - 1));
  return values[index];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6 && argc != 7) {
      throw std::runtime_error(
          "usage: perplexity REFERENCE CANDIDATE TOKENIZER TEXT CHUNKS [PER_TOKEN_TSV]");
    }
    const int chunks = std::stoi(argv[5]);
    unremarkable::Engine reference(argv[1], kContext);
    unremarkable::Engine candidate(argv[2], kContext);
    const int vocabulary = reference.config().vocab_size;
    if (candidate.config().vocab_size != vocabulary) {
      throw std::runtime_error("reference and candidate vocabularies differ");
    }
    unremarkable::Tokenizer tokenizer(argv[3], vocabulary);
    std::ifstream file(argv[4], std::ios::binary);
    if (!file) {
      throw std::runtime_error(std::string("cannot open text: ") + argv[4]);
    }
    std::stringstream text;
    text << file.rdbuf();
    const std::vector<int> tokens = tokenizer.encode(text.str());
    const int available = static_cast<int>(tokens.size() / kContext);
    if (chunks <= 0 || chunks > available) {
      throw std::runtime_error("text holds only " + std::to_string(available) + " chunks");
    }
    std::ofstream rows;
    if (argc == 7) {
      rows.open(argv[6]);
      rows << "chunk\tposition\tnext\treference_nll\tcandidate_nll\tkl\tsame_top\n";
    }

    // Per token, and per chunk: neighbouring tokens are correlated, so chunk
    // means give the honest standard error.
    std::vector<double> reference_nll, candidate_nll, token_difference, token_kl;
    std::vector<double> chunk_difference, chunk_kl;
    std::vector<double> reference_log, candidate_log;
    int same_top = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int chunk = 0; chunk < chunks; ++chunk) {
      reference.reset();
      candidate.reset();
      const int* ids = tokens.data() + static_cast<size_t>(chunk) * kContext;
      double difference_sum = 0, kl_sum = 0;
      int scored = 0;
      for (int position = 0; position + 1 < kContext; ++position) {
        const std::span<const float> reference_logits = reference.forward(ids[position], position);
        const std::span<const float> candidate_logits = candidate.forward(ids[position], position);
        // Score only the second half, so every prediction has 256+ tokens of context.
        if (position < kContext / 2) {
          continue;
        }
        const int reference_top = log_softmax(reference_logits, reference_log);
        const int candidate_top = log_softmax(candidate_logits, candidate_log);
        double kl = 0;
        for (int i = 0; i < vocabulary; ++i) {
          kl += std::exp(reference_log[i]) * (reference_log[i] - candidate_log[i]);
        }
        const int next = ids[position + 1];
        const double reference_loss = -reference_log[next];
        const double candidate_loss = -candidate_log[next];
        reference_nll.push_back(reference_loss);
        candidate_nll.push_back(candidate_loss);
        token_difference.push_back(candidate_loss - reference_loss);
        token_kl.push_back(kl);
        same_top += reference_top == candidate_top;
        difference_sum += candidate_loss - reference_loss;
        kl_sum += kl;
        ++scored;
        if (rows) {
          rows << chunk << '\t' << position << '\t' << next << '\t' << reference_loss << '\t'
               << candidate_loss << '\t' << kl << '\t' << (reference_top == candidate_top) << '\n';
        }
      }
      chunk_difference.push_back(difference_sum / scored);
      chunk_kl.push_back(kl_sum / scored);
      std::cerr << "chunk " << chunk + 1 << "/" << chunks << "\n";
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const Estimate reference_loss = estimate(reference_nll);
    const Estimate candidate_loss = estimate(candidate_nll);
    const Estimate difference = estimate(chunk_difference);
    const Estimate difference_by_token = estimate(token_difference);
    const Estimate kl = estimate(chunk_kl);
    const size_t scored = token_kl.size();
    std::cout.precision(6);
    std::cout << "{\"chunks\":" << chunks << ",\"context\":" << kContext
              << ",\"tokens_in_text\":" << tokens.size() << ",\"scored\":" << scored
              << ",\"reference_ppl\":" << std::exp(reference_loss.mean)
              << ",\"candidate_ppl\":" << std::exp(candidate_loss.mean)
              << ",\"nll_difference\":" << difference.mean
              << ",\"nll_difference_se_chunks\":" << difference.standard_error
              << ",\"nll_difference_se_tokens\":" << difference_by_token.standard_error
              << ",\"ppl_ratio\":" << std::exp(difference.mean) << ",\"kl_mean\":" << kl.mean
              << ",\"kl_se_chunks\":" << kl.standard_error
              << ",\"kl_median\":" << percentile(token_kl, 0.5)
              << ",\"kl_p99\":" << percentile(token_kl, 0.99)
              << ",\"kl_max\":" << *std::max_element(token_kl.begin(), token_kl.end())
              << ",\"same_top\":" << static_cast<double>(same_top) / scored
              << ",\"seconds\":" << seconds << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
