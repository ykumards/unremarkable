#include "pool.h"

#include <algorithm>

namespace unremarkable {
namespace {

// Chunk boundaries that differ by at most one row, so a short final chunk never
// leaves one core idle while another finishes a long one.
int boundary(int count, int part, int parts) {
  const long long scaled = static_cast<long long>(count) * part;
  return static_cast<int>(scaled / parts);
}

}  // namespace

Pool::Pool(int threads) : threads_(std::max(1, threads)) {
  for (int i = 1; i < threads_; i++) {
    workers_.emplace_back([this, i] { worker(i); });
  }
}

Pool::~Pool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  for (std::thread& thread : workers_) {
    thread.join();
  }
}

void Pool::worker(int index) {
  uint64_t seen = 0;
  for (;;) {
    std::unique_lock<std::mutex> lock(mutex_);
    wake_.wait(lock, [this, &seen] { return stopping_ || generation_ != seen; });
    if (stopping_) {
      return;
    }
    seen = generation_;
    const std::function<void(int, int)>* work = work_;
    const int begin = boundary(count_, index, threads_);
    const int end = boundary(count_, index + 1, threads_);
    lock.unlock();
    if (begin < end) {
      (*work)(begin, end);
    }
    lock.lock();
    if (--outstanding_ == 0) {
      done_.notify_one();
    }
  }
}

void Pool::run(int count, int minimum, const std::function<void(int, int)>& work) {
  if (threads_ == 1 || count < minimum) {
    if (count > 0) {
      work(0, count);
    }
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    work_ = &work;
    count_ = count;
    outstanding_ = threads_ - 1;
    generation_++;
  }
  wake_.notify_all();
  // The calling thread takes the first chunk rather than idling.
  const int end = boundary(count, 1, threads_);
  if (end > 0) {
    work(0, end);
  }
  std::unique_lock<std::mutex> lock(mutex_);
  done_.wait(lock, [this] { return outstanding_ == 0; });
}

}  // namespace unremarkable
