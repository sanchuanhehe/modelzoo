/*
 * Copyright (c) ModelZoo. 2025-2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "log.h"

namespace Infer {
std::atomic<LogLevel> Logger::globalLevel_{INFO};
std::mutex Logger::mutex_;

Logger::Logger(LogLevel level, const char* file, int line)
        : level_(level), file_(file), line_(line) {}

Logger::~Logger()
{
    if (level_ < globalLevel_) return;
    std::string formatted = "[" + levelToString() + "] " + formatTime() + " ["
                            + GetFileName(file_) + ":" + std::to_string(line_)
                            + "] " + stream_.str();
    outputLog(formatted);
}

LogStream& Logger::stream()
{ 
    return stream_;
}

void Logger::setLogLevel(LogLevel level)
{ 
    globalLevel_.store(level); 
}

std::string Logger::formatTime() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::GetFileName(const char* fullPath) {
    std::string path(fullPath);
    size_t pos = path.find_last_of("/\\");
    return (pos != std::string::npos) ? path.substr(pos + 1) : path;
}

std::string Logger::levelToString() {
    switch(level_) {
        case DEBUG:   return "DEBUG";
        case INFO:    return "INFO";
        case WARNING: return "WARN";
        case ERROR:   return "ERROR";
        default:      return "UNKNOWN";
    }
}

void Logger::outputLog(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << msg << std::endl;
}
}