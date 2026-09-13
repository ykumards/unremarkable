#include "worker.h"

#include <stdexcept>

#ifdef UNREMARKABLE_PROFILE
#include <chrono>
#endif

namespace unremarkable {

#ifdef UNREMARKABLE_PROFILE
namespace {
double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace
#endif

RowWorker::RowWorker(int threads) : threads_(threads) {
  if (threads != 1 && threads != 2) {
    throw std::runtime_error("threads must be 1 or 2");
  }
  if (threads == 2) {
    worker_ = std::thread([this] { loop(); });
  }
}

RowWorker::~RowWorker() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void RowWorker::loop() {
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [this] { return stopping_ || pending_; });
    if (stopping_) {
      return;
    }
    const auto* work = work_;
    const int begin = begin_;
    const int end = end_;
    lock.unlock();
#ifdef UNREMARKABLE_PROFILE
    const double started = now();
#endif
    (*work)(begin, end);
    lock.lock();
#ifdef UNREMARKABLE_PROFILE
    busy_ += now() - started;
#endif
    pending_ = false;
    done_.notify_one();
  }
}

void RowWorker::run(int rows, const std::function<void(int, int)>& work,
                    int minimum_parallel_rows) {
  if (threads_ == 1 || rows < minimum_parallel_rows) {
    if (rows > 0) {
      work(0, rows);
    }
    return;
  }
  const int middle = rows / 2;
  {
    std::lock_guard lock(mutex_);
    work_ = &work;
    begin_ = middle;
    end_ = rows;
    pending_ = true;
  }
  wake_.notify_one();
  work(0, middle);
#ifdef UNREMARKABLE_PROFILE
  const double finished = now();
#endif
  std::unique_lock lock(mutex_);
  done_.wait(lock, [this] { return !pending_; });
#ifdef UNREMARKABLE_PROFILE
  waiting_ += now() - finished;
#endif
  work_ = nullptr;
}

}  // namespace unremarkable
