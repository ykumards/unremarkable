#include "worker.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
  for (int threads : {1, 2}) {
    for (int repeat = 0; repeat < 4; ++repeat) {
      unremarkable::RowWorker worker(threads);
      for (int count : {0, 1, 63, 64, 65, 127, 512}) {
        std::vector<std::atomic<int>> hits(count);
        for (int job = 0; job < 100; ++job) {
          worker.run(count, [&](int begin, int end) {
            for (int row = begin; row < end; ++row) {
              ++hits.at(row);
            }
          });
          for (const auto& hit : hits) {
            if (hit.load() != job + 1) {
              std::fputs("row skipped, duplicated, or returned before completion\n", stderr);
              return 1;
            }
          }
        }
      }
    }
  }
  for (int threads : {0, 3}) {
    try {
      unremarkable::RowWorker worker(threads);
      return 1;
    } catch (const std::runtime_error&) {
    }
  }
  std::puts("worker: row coverage, repeated jobs, shutdown, and thread limits passed");
}
