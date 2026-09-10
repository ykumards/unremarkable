#ifndef UNREMARKABLE_SRC_POOL_H_
#define UNREMARKABLE_SRC_POOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace unremarkable {

// Splits a row range across a fixed set of workers. A forward pass issues a few
// hundred of these per token, so the workers persist and are woken rather than
// created; the caller runs the first chunk itself and joins the rest.
class Pool {
 public:
  explicit Pool(int threads);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  int threads() const { return threads_; }

  // Calls work(begin, end) over disjoint, contiguous chunks covering [0, count).
  // Chunks are whole rows, so each output element is produced by exactly one
  // thread in the same order it would be alone: results do not depend on how
  // many threads run. Below `minimum` rows the split costs more than it saves
  // and the caller does the whole range itself.
  void run(int count, int minimum, const std::function<void(int, int)>& work);

 private:
  void worker(int index);

  const int threads_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  const std::function<void(int, int)>* work_ = nullptr;
  int count_ = 0;
  uint64_t generation_ = 0;
  int outstanding_ = 0;
  bool stopping_ = false;
};

}  // namespace unremarkable
#endif
