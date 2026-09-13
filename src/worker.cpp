#include "worker.h"

#include <stdexcept>

namespace unremarkable {

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
    (*work)(begin, end);
    lock.lock();
    pending_ = false;
    done_.notify_one();
  }
}

void RowWorker::run(int rows, const std::function<void(int, int)>& work) {
  constexpr int minimum_parallel_rows = 64;
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
  std::unique_lock lock(mutex_);
  done_.wait(lock, [this] { return !pending_; });
  work_ = nullptr;
}

}  // namespace unremarkable
