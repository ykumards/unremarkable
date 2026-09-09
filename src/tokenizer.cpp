// Legacy BPE adapted from karpathy/llama2.c; byte-level BPE is local. See THIRD_PARTY.md.
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace unremarkable {
namespace {

constexpr uint32_t kMagic = 0x4B4F5455;  // "UTOK"; legacy files open with a small length.
constexpr int32_t kVersion = 1;

// ASCII-exact classes for the byte-level pre-tokenizer. Bytes above ASCII are
// treated as letters, which matches the reference split for Latin and CJK text
// but not for symbols such as emoji, where a chunk boundary may differ.
bool is_space_byte(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}
bool is_letter_byte(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80;
}
bool is_digit_byte(unsigned char c) { return c >= '0' && c <= '9'; }

}  // namespace

Tokenizer::Tokenizer(const std::string& path, int vocabulary) {
  if (vocabulary < 259) {
    throw std::runtime_error("invalid tokenizer vocabulary");
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open tokenizer: " + path);
  }
  uint32_t magic = 0;
  if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
    throw std::runtime_error("truncated or unreadable tokenizer");
  }
  byte_level_ = magic == kMagic;
  file.seekg(0);
  vocabulary_.resize(vocabulary);
  scores_.resize(vocabulary);
  special_.assign(vocabulary, 0);
  if (byte_level_) {
    load_byte_level(file, vocabulary);
  } else {
    load_legacy(file, vocabulary);
  }
  if (file.peek() != std::char_traits<char>::eof() || file.bad()) {
    throw std::runtime_error("tokenizer vocabulary size mismatch");
  }
  index_vocabulary();
}

void Tokenizer::load_legacy(std::ifstream& file, int vocabulary) {
  auto read = [&](void* output, size_t bytes) {
    if (!file.read(static_cast<char*>(output), static_cast<std::streamsize>(bytes))) {
      throw std::runtime_error("truncated or unreadable tokenizer");
    }
  };
  uint32_t maximum_length;
  read(&maximum_length, 4);
  if (!maximum_length || maximum_length > 65536) {
    throw std::runtime_error("invalid tokenizer maximum token length");
  }
  for (int i = 0; i < vocabulary; i++) {
    int32_t length;
    read(&scores_[i], 4);
    read(&length, 4);
    if (length <= 0 || static_cast<uint32_t>(length) > maximum_length ||
        !std::isfinite(scores_[i])) {
      throw std::runtime_error("invalid tokenizer entry");
    }
    vocabulary_[i].resize(length);
    read(vocabulary_[i].data(), length);
    if (vocabulary_[i].find('\0') != std::string::npos) {
      throw std::runtime_error("tokenizer entry contains NUL");
    }
  }
  for (int i = 0; i < 256; i++) {
    char expected[7];
    std::snprintf(expected, sizeof(expected), "<0x%02X>", i);
    if (vocabulary_[i + 3] != expected) {
      throw std::runtime_error("tokenizer must use byte fallback IDs 3..258");
    }
    bytes_[i] = static_cast<char>(i);
  }
}

void Tokenizer::load_byte_level(std::ifstream& file, int vocabulary) {
  auto read = [&](void* output, size_t bytes) {
    if (!file.read(static_cast<char*>(output), static_cast<std::streamsize>(bytes))) {
      throw std::runtime_error("truncated or unreadable tokenizer");
    }
  };
  uint32_t magic;
  int32_t version, count;
  uint32_t maximum_length;
  read(&magic, 4);
  read(&version, 4);
  read(&count, 4);
  read(&maximum_length, 4);
  if (version != kVersion) {
    throw std::runtime_error("unsupported tokenizer version");
  }
  if (count != vocabulary) {
    throw std::runtime_error("tokenizer vocabulary size mismatch");
  }
  if (!maximum_length || maximum_length > 65536) {
    throw std::runtime_error("invalid tokenizer maximum token length");
  }
  for (int i = 0; i < vocabulary; i++) {
    int32_t length;
    uint8_t flags;
    read(&scores_[i], 4);
    read(&length, 4);
    read(&flags, 1);
    if (length <= 0 || static_cast<uint32_t>(length) > maximum_length ||
        !std::isfinite(scores_[i]) || flags > 1) {
      throw std::runtime_error("invalid tokenizer entry");
    }
    vocabulary_[i].resize(length);
    read(vocabulary_[i].data(), length);
    special_[i] = flags;
  }
  byte_token_.fill(-1);
  for (int i = 0; i < vocabulary; i++) {
    if (!special_[i] && vocabulary_[i].size() == 1) {
      byte_token_[static_cast<unsigned char>(vocabulary_[i][0])] = i;
    }
  }
}

void Tokenizer::index_vocabulary() {
  sorted_ids_.clear();
  sorted_ids_.reserve(vocabulary_.size());
  for (size_t i = 0; i < vocabulary_.size(); i++) {
    if (!special_[i]) {
      sorted_ids_.push_back(static_cast<int>(i));
    }
  }
  std::sort(sorted_ids_.begin(), sorted_ids_.end(),
            [&](int a, int b) { return vocabulary_[a] < vocabulary_[b]; });
  for (size_t i = 1; i < sorted_ids_.size(); i++) {
    if (vocabulary_[sorted_ids_[i - 1]] == vocabulary_[sorted_ids_[i]]) {
      throw std::runtime_error("duplicate tokenizer entry");
    }
  }
  if (!byte_level_ && lookup(" ") < 0) {
    throw std::runtime_error("tokenizer is missing space token");
  }
}

int Tokenizer::lookup(std::string_view text) const {
  auto found =
      std::lower_bound(sorted_ids_.begin(), sorted_ids_.end(), text,
                       [&](int id, std::string_view value) { return vocabulary_[id] < value; });
  return found != sorted_ids_.end() && vocabulary_[*found] == text ? *found : -1;
}

int Tokenizer::special_id(std::string_view text) const {
  for (size_t i = 0; i < vocabulary_.size(); i++) {
    if (special_[i] && vocabulary_[i] == text) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Repeatedly merge the highest-scoring adjacent pair in tokens[begin..end).
// Deliberately simple.
void Tokenizer::merge_range(std::vector<int>& tokens, size_t begin) const {
  while (tokens.size() - begin > 1) {
    float best_score = -1e30f;
    int best_id = -1;
    size_t best_index = 0;
    for (size_t i = begin; i + 1 < tokens.size(); i++) {
      int id = lookup(vocabulary_[tokens[i]] + vocabulary_[tokens[i + 1]]);
      if (id >= 0 && scores_[id] > best_score) {
        best_score = scores_[id];
        best_id = id;
        best_index = i;
      }
    }
    if (best_id < 0) {
      break;
    }
    tokens[best_index] = best_id;
    tokens.erase(tokens.begin() + static_cast<ptrdiff_t>(best_index) + 1);
  }
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
  std::vector<int> tokens;
  encode_into(text, tokens);
  return tokens;
}

void Tokenizer::encode_into(std::string_view text, std::vector<int>& tokens) const {
  if (text.size() > static_cast<size_t>(std::numeric_limits<int>::max()) - 3) {
    throw std::runtime_error("prompt is too large");
  }
  if (!byte_level_) {
    if (text.find('\0') != std::string_view::npos) {
      throw std::runtime_error("prompt contains NUL");
    }
    tokens.push_back(1);  // Beginning of sequence (BOS).
    if (!text.empty()) {
      tokens.push_back(lookup(" "));  // SentencePiece prefix.
    }
    std::string codepoint;
    for (size_t i = 0; i < text.size(); i++) {
      const auto byte = static_cast<unsigned char>(text[i]);
      if ((byte & 0xC0) != 0x80) {
        codepoint.clear();
      }
      codepoint += text[i];
      if (i + 1 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80 &&
          codepoint.size() < 4) {
        continue;
      }
      int id = lookup(codepoint);
      if (id >= 0) {
        tokens.push_back(id);
      } else {
        for (unsigned char c : codepoint) {
          tokens.push_back(c + 3);
        }
      }
      codepoint.clear();
    }
    merge_range(tokens, 0);
    return;
  }

  // Byte-level: split on the reference's word boundaries, then merge inside each
  // chunk so a merge can never span two chunks.
  static constexpr std::string_view kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  const size_t size = text.size();
  size_t i = 0;
  while (i < size) {
    size_t end = 0;
    for (std::string_view contraction : kContractions) {
      if (text.compare(i, contraction.size(), contraction) == 0) {
        end = i + contraction.size();
        break;
      }
    }
    if (!end) {
      const size_t space = (text[i] == ' ' && i + 1 < size) ? 1 : 0;
      size_t j = i + space;
      const auto at = [&](size_t k) { return static_cast<unsigned char>(text[k]); };
      if (j < size && is_letter_byte(at(j))) {
        while (j < size && is_letter_byte(at(j))) {
          j++;
        }
        end = j;
      } else if (j < size && is_digit_byte(at(j))) {
        while (j < size && is_digit_byte(at(j))) {
          j++;
        }
        end = j;
      } else if (j < size && !is_space_byte(at(j))) {
        while (j < size && !is_space_byte(at(j)) && !is_letter_byte(at(j)) &&
               !is_digit_byte(at(j))) {
          j++;
        }
        end = j;
      } else {
        j = i;
        while (j < size && is_space_byte(at(j))) {
          j++;
        }
        // A whitespace run that continues into text leaves its last byte behind,
        // matching the reference's trailing-whitespace lookahead.
        if (j < size && j - i > 1) {
          j--;
        }
        end = j == i ? i + 1 : j;
      }
    }
    const size_t begin = tokens.size();
    for (size_t k = i; k < end; k++) {
      const int id = byte_token_[static_cast<unsigned char>(text[k])];
      if (id < 0) {
        throw std::runtime_error("prompt contains a byte the tokenizer cannot represent");
      }
      tokens.push_back(id);
    }
    merge_range(tokens, begin);
    i = end;
  }
}

std::string_view Tokenizer::decode(int previous, int token) const {
  if (token < 0 || static_cast<size_t>(token) >= vocabulary_.size()) {
    throw std::runtime_error("token ID out of range");
  }
  if (byte_level_) {
    return vocabulary_[token];
  }
  if (token >= 3 && token <= 258) {
    return {&bytes_[token - 3], 1};
  }
  std::string_view piece = vocabulary_[token];
  if (previous == 1 && piece.starts_with(' ')) {
    piece.remove_prefix(1);
  }
  return piece;
}

}  // namespace unremarkable
