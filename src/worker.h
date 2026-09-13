#ifndef UNREMARKABLE_SRC_WORKER_H_
#define UNREMARKABLE_SRC_WORKER_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace unremarkable {

// One optional background thread. The caller computes the first half of the rows.
class RowWorker {
 public:
  explicit RowWorker(int threads);
  ~RowWorker();
  RowWorker(const RowWorker&) = delete;
  RowWorker& operator=(const RowWorker&) = delete;

  int threads() const { return threads_; }

  // Run disjoint ranges covering [0, rows), then wait for completion.
  // Calls must not overlap, and work must not throw. Small jobs stay on the caller.
  void run(int rows, const std::function<void(int, int)>& work);

#ifdef UNREMARKABLE_PROFILE
  // Elapsed seconds, including lock acquisition. Read after run() returns.
  double waiting_seconds() const { return waiting_; }
  double busy_seconds() const { return busy_; }
#endif

 private:
  void loop();
#ifdef UNREMARKABLE_PROFILE
  double waiting_ = 0;
  double busy_ = 0;
#endif

  int threads_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  const std::function<void(int, int)>* work_ = nullptr;
  int begin_ = 0;
  int end_ = 0;
  bool pending_ = false;
  bool stopping_ = false;
  std::thread worker_;
};

}  // namespace unremarkable
#endif
