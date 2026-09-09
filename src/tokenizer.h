#ifndef UNREMARKABLE_SRC_TOKENIZER_H_
#define UNREMARKABLE_SRC_TOKENIZER_H_

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace unremarkable {

// Two on-disk vocabularies share one merge loop:
//   * the legacy llama2.c SentencePiece export, with byte fallback IDs 3..258;
//   * the byte-level BPE export written by tools/export_hf.py, whose entries
//     hold raw bytes and whose scores are negated merge ranks.
// Byte-level encoding never emits a special token, so prompt text cannot forge
// a chat role marker; callers insert those by identifier.
class Tokenizer {
 public:
  Tokenizer(const std::string& path, int vocabulary);
  bool byte_level() const { return byte_level_; }

  // Legacy vocabularies prepend BOS and a SentencePiece space prefix; byte-level
  // vocabularies encode the text exactly.
  std::vector<int> encode(std::string_view text) const;
  void encode_into(std::string_view text, std::vector<int>& tokens) const;
  std::string_view decode(int previous, int token) const;

  // Identifier of a special token such as "<|im_start|>", or -1 when absent.
  int special_id(std::string_view text) const;

 private:
  void load_legacy(std::ifstream& file, int vocabulary);
  void load_byte_level(std::ifstream& file, int vocabulary);
  void index_vocabulary();
  int lookup(std::string_view text) const;
  void merge_range(std::vector<int>& tokens, size_t begin) const;

  std::vector<std::string> vocabulary_;
  std::vector<float> scores_;
  std::vector<int> sorted_ids_;        // Mergeable entries only, ordered by text.
  std::vector<uint8_t> special_;       // Excluded from lookup, so text cannot forge one.
  std::array<char, 256> bytes_{};      // Legacy byte-fallback payloads.
  std::array<int, 256> byte_token_{};  // Byte-level base tokens; -1 when absent.
  bool byte_level_ = false;
};

}  // namespace unremarkable
#endif
