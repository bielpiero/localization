#ifndef TIMING_LOGGER_HPP
#define TIMING_LOGGER_HPP

#include <chrono>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

class TimingLogger {
public:
  static double nowMs() {
    using clock = std::chrono::steady_clock;

    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
  }

  static void log(const std::string &operation, double duration_ms) {
    static std::mutex mutex;
    static std::ofstream file("/home/sara/localization_timing.csv",
                              std::ios::out);

    static bool header_written = false;

    std::lock_guard<std::mutex> lock(mutex);

    if (!header_written) {
      file << "operation,duration_ms\n";
      header_written = true;
    }

    file << operation << "," << std::fixed << std::setprecision(9)
         << duration_ms << "\n";
  }
};
#endif