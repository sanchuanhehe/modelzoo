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
#pragma once

#include <iostream>
#include <sstream>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <chrono>
#include <string>

namespace Infer {
#define LOG(level) Logger(LogLevel::level, __FILE__, __LINE__).stream()
#define ERROR_LOG(fmt, ...)fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)

enum LogLevel { DEBUG, INFO, WARNING, ERROR };

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

class LogStream {
public:
    template <typename T>
    LogStream& operator<<(const T& value) {
        buffer_ << value;
        return *this;
    }

    std::string str() const { 
        return buffer_.str(); 
    }

private:
    std::ostringstream buffer_;
};

class Logger {
public:
    Logger(LogLevel level, const char* file, int line);
    
    ~Logger();

    LogStream& stream();

    static void setLogLevel(LogLevel level);

private:
    std::string formatTime();
    std::string GetFileName(const char* fullPath);
    std::string levelToString();
    void outputLog(const std::string& msg);

    LogLevel level_;
    const char* file_;
    int line_;
    LogStream stream_;
    static std::atomic<LogLevel> globalLevel_;
    static std::mutex mutex_;
};
}