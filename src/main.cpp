#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif

#include <sys/resource.h>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "engine.h"
#include "tokenizer.h"

namespace {

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void print_piece(std::string_view text) {
  for (unsigned char byte : text) {
    if ((byte >= 0x20 && byte != 0x7f) || byte == '\n' || byte == '\t') {
      std::cout.put(static_cast<char>(byte));
    }
  }
}

struct Options {
  std::string tokenizer = "models/tokenizer.bin";
  std::string prompt;
  std::string system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
  std::string opening;
  int limit = 256;
  int context = 0;
  int seed = 1;
  float temperature = 0;
  float top_p = 0.9f;
};

int positive_int(const char* text) {
  char* end;
  errno = 0;
  long value = std::strtol(text, &end, 10);
  if (errno || end == text || *end || value <= 0 || value > INT_MAX) {
    throw std::runtime_error("expected a positive integer no greater than INT_MAX");
  }
  return static_cast<int>(value);
}

float nonnegative_float(const char* text) {
  char* end;
  errno = 0;
  float value = std::strtof(text, &end);
  if (errno || end == text || *end || !std::isfinite(value) || value < 0) {
    throw std::runtime_error("expected a finite nonnegative number");
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 2; i < argc; i += 2) {
    if (i + 1 == argc) {
      throw std::runtime_error("missing option value");
    }
    std::string_view flag = argv[i];
    const char* value = argv[i + 1];
    if (flag == "-z") {
      options.tokenizer = value;
    } else if (flag == "-i") {
      options.prompt = value;
    } else if (flag == "-y") {
      options.system = value;
    } else if (flag == "-a") {
      options.opening = value;
    } else if (flag == "-n") {
      options.limit = positive_int(value);
    } else if (flag == "-c") {
      options.context = positive_int(value);
    } else if (flag == "-s") {
      options.seed = positive_int(value);
    } else if (flag == "-t") {
      options.temperature = nonnegative_float(value);
    } else if (flag == "-p") {
      options.top_p = nonnegative_float(value);
    } else {
      throw std::runtime_error("unknown option (use --help)");
    }
  }
  if (options.top_p > 1) {
    throw std::runtime_error("top-p must be between 0 and 1");
  }
  return options;
}

void generate(unremarkable::Engine& engine, const unremarkable::Tokenizer& tokenizer,
              unremarkable::Sampler& sampler, const Options& options, double load_seconds) {
  double start = now();
  // A vocabulary carrying the ChatML markers gets the instruct template; the
  // legacy TinyStories vocabularies continue the prompt text directly.
  const int im_start = tokenizer.special_id("<|im_start|>");
  const int im_end = tokenizer.special_id("<|im_end|>");
  const int end_of_text = tokenizer.special_id("<|endoftext|>");
  const bool chat = im_start >= 0 && im_end >= 0;
  std::vector<int> tokens;
  if (chat) {
    auto turn = [&](std::string_view role, std::string_view content) {
      tokens.push_back(im_start);
      tokenizer.encode_into(std::string(role) + "\n" + std::string(content), tokens);
      tokens.push_back(im_end);
      tokenizer.encode_into("\n", tokens);
    };
    if (!options.system.empty()) {
      turn("system", options.system);
    }
    turn("user", options.prompt);
    tokens.push_back(im_start);
    tokenizer.encode_into("assistant\n", tokens);
    // Opening the reply for the model leaves it only the continuation to write,
    // which a small model does far more reliably than starting from nothing.
    if (!options.opening.empty()) {
      tokenizer.encode_into(options.opening, tokens);
    }
  } else {
    tokenizer.encode_into(options.prompt, tokens);
  }
  const int prompt_tokens = static_cast<int>(tokens.size());
  if (prompt_tokens > engine.config().seq_len) {
    throw std::runtime_error("prompt exceeds context capacity");
  }
  if (!chat) {
    print_piece(options.prompt);
    std::cout.flush();
  } else if (!options.opening.empty()) {
    print_piece(options.opening);
    std::cout.flush();
  }

  double prefill_start = now();
  std::span<float> logits;
  for (int pos = 0; pos < prompt_tokens; pos++) {
    logits = engine.forward(tokens[pos], pos);
  }
  double prefill_seconds = now() - prefill_start;
  int previous = tokens.back();
  int pos = prompt_tokens, generated = 0, decode_tokens = 0;
  double decode_seconds = 0, ttft_seconds = 0;
  const char* stop = "limit";

  while (generated < options.limit) {
    double step_start = now();
    if (generated) {
      if (pos >= engine.config().seq_len) {
        stop = "context";
        break;
      }
      logits = engine.forward(previous, pos++);
    }
    int token = sampler.sample(logits);
    double step_end = now();
    // Chat vocabularies close a turn with <|im_end|>; TinyStories reuses BOS as a
    // sequence delimiter and also honors ordinary EOS.
    if (chat ? (token == im_end || token == end_of_text) : (token == 1 || token == 2)) {
      stop = chat ? "turn" : (token == 1 ? "bos" : "eos");
      break;
    }
    if (!generated) {
      ttft_seconds = step_end - start;
    } else {
      decode_tokens++;
      decode_seconds += step_end - step_start;
    }
    print_piece(tokenizer.decode(previous, token));
    std::cout.flush();
    previous = token;
    generated++;
  }
  std::cout << '\n';
  std::cout.flush();
  if (!std::cout) {
    throw std::runtime_error("failed to write generated text");
  }
  double elapsed = now() - start;
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage)) {
    throw std::runtime_error("getrusage failed");
  }
#ifdef __APPLE__
  double rss_mib = usage.ru_maxrss / 1048576.0;
#else
  double rss_mib = usage.ru_maxrss / 1024.0;
#endif
  std::fprintf(stderr,
               "{\"prompt_tokens\":%d,\"generated_tokens\":%d,\"decode_tokens\":%d,"
               "\"context\":%d,\"stop\":\"%s\",\"load_ms\":%.3f,\"prefill_ms\":%.3f,"
               "\"ttft_ms\":%.3f,\"decode_ms\":%.3f,\"decode_tok_s\":%.3f,"
               "\"generation_ms\":%.3f,\"weights_mib\":%.3f,\"kv_cache_mib\":%.3f,"
               "\"peak_rss_mib\":%.3f}\n",
               prompt_tokens, generated, decode_tokens, engine.config().seq_len, stop,
               load_seconds * 1000, prefill_seconds * 1000, ttft_seconds * 1000,
               decode_seconds * 1000, decode_seconds > 0 ? decode_tokens / decode_seconds : 0,
               elapsed * 1000, engine.weight_bytes() / 1048576.0, engine.kv_bytes() / 1048576.0,
               rss_mib);
}

void usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s MODEL.bin [-z TOKENIZER.bin] [-i PROMPT] [options]\n"
               "  -n N      maximum new tokens, excluding prompt (default 256)\n"
               "  -c N      context capacity, at most model maximum (default model "
               "maximum)\n"
               "  -t TEMP   temperature; 0 means greedy (default 0)\n"
               "  -p TOP_P  nucleus probability; 0 or 1 disables filtering (default "
               "0.9)\n"
               "  -s SEED   positive random seed (default 1)\n"
               "  -z PATH   tokenizer (default models/tokenizer.bin)\n"
               "  -i TEXT   prompt (default empty)\n"
               "  -a TEXT   open the reply with this text and continue it; chat\n"
               "            templates only, and it is printed with the reply\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    usage(argv[0]);
    return 0;
  }
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }
  try {
    const Options options = parse_options(argc, argv);
    double start = now();
    unremarkable::Engine engine(argv[1], options.context);
    double load_seconds = now() - start;
    unremarkable::Tokenizer tokenizer(options.tokenizer, engine.config().vocab_size);
    unremarkable::Sampler sampler(engine.config().vocab_size, options.temperature, options.top_p,
                                  options.seed);
    generate(engine, tokenizer, sampler, options, load_seconds);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
