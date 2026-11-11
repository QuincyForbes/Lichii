// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>

#if defined(__ANDROID__)
#include <android/log.h>
#endif
#ifdef _WIN32
#include <windows.h> // For OutputDebugStringW
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <boost/algorithm/string/replace.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/literals.h"
#include "common/polyfill_thread.h"
#include "common/scm_rev.h"
#include "common/thread.h"

#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "common/bounded_threadsafe_queue.h"

namespace Common::Log {

/// @brief A log entry. Log entries are store in a structured format to permit more varied output
/// formatting on different frontends, as well as facilitating filtering and aggregation.
struct Entry {
    char const* message = nullptr;
    size_t message_len = 0;
    std::chrono::microseconds timestamp;
    Class log_class{};
    Level log_level{};
    const char* filename = nullptr;
    const char* function = nullptr;
    uint32_t line_num = 0;
};

namespace {

/// @brief Returns the name of the passed log class as a C-string. Subclasses are separated by periods
/// instead of underscores as in the enumeration.
/// @note GetClassName is a macro defined by Windows.h, grrr...
const char* GetLogClassName(Class log_class) {
    switch (log_class) {
#define CLS(x) case Class::x: return #x;
#define SUB(x, y) case Class::x##_##y: return #x "." #y;
#include "common/log_classes.inc"
#undef CLS
#undef SUB
    default: return "?";
    }
}

/// @brief Returns the name of the passed log level as a C-string.
const char* GetLevelName(Level log_level) {
    switch (log_level) {
#define LVL(x) case Level::x: return #x;
    LVL(Trace)
    LVL(Debug)
    LVL(Info)
    LVL(Warning)
    LVL(Error)
    LVL(Critical)
#undef LVL
    default: return "?";
    }
}

template <typename It>
Level GetLevelByName(const It begin, const It end) {
    for (u32 i = 0; i < u32(Level::Count); ++i) {
        const char* level_name = GetLevelName(Level(i));
        if (Common::ComparePartialString(begin, end, level_name))
            return Level(i);
    }
    return Level::Count;
}

template <typename It>
Class GetClassByName(const It begin, const It end) {
    for (u32 i = 0; i < u32(Class::Count); ++i) {
        const char* level_name = GetLogClassName(Class(i));
        if (Common::ComparePartialString(begin, end, level_name))
            return Class(i);
    }
    return Class::Count;
}

template <typename Iterator>
bool ParseFilterRule(Filter& instance, Iterator begin, Iterator end) {
    auto level_separator = std::find(begin, end, ':');
    if (level_separator == end) {
        LOG_ERROR(Log, "Invalid log filter. Must specify a log level after `:`: {}", std::string(begin, end));
        return false;
    }
    const Level level = GetLevelByName(level_separator + 1, end);
    if (level == Level::Count) {
        LOG_ERROR(Log, "Unknown log level in filter: {}", std::string(begin, end));
        return false;
    }
    if (Common::ComparePartialString(begin, level_separator, "*")) {
        instance.class_levels.fill(level);
        return true;
    }
    const Class log_class = GetClassByName(begin, level_separator);
    if (log_class == Class::Count) {
        LOG_ERROR(Log, "Unknown log class in filter: {}", std::string(begin, end));
        return false;
    }
    instance.SetClassLevel(log_class, level);
    return true;
}

/// @brief Trims up to and including the last of ../, ..\, src/, src\ in a string
/// do not be fooled this isn't generating new strings on .rodata :)
constexpr const char* TrimSourcePath(std::string_view source) noexcept {
    const auto rfind = [source](const std::string_view match) {
        return source.rfind(match) == source.npos ? 0 : (source.rfind(match) + match.size());
    };
    auto idx = (std::max)({rfind("src/"), rfind("src\\"), rfind("../"), rfind("..\\")});
    return source.data() + idx;
}

/// @brief Interface for logging backends.
struct Backend {
    virtual ~Backend() noexcept = default;
    virtual void Write(const Entry& entry, std::string_view formatted_line) noexcept = 0;
    virtual void Flush() noexcept = 0;
};

/// @brief Backend that writes to stdout and with color
struct ColorConsoleBackend final : public Backend {
#ifdef _WIN32
    explicit ColorConsoleBackend() noexcept {
        console_handle = GetStdHandle(STD_ERROR_HANDLE);
        GetConsoleScreenBufferInfo(console_handle, &original_info);
    }
    ~ColorConsoleBackend() noexcept override {
        SetConsoleTextAttribute(console_handle, original_info.wAttributes);
    }
    void Write(const Entry& entry, std::string_view formatted_line) noexcept override {
        if (enabled.load(std::memory_order_relaxed) && console_handle != INVALID_HANDLE_VALUE) {
            WORD color = WORD([&entry]() {
                switch (entry.log_level) {
                case Level::Debug: return FOREGROUND_GREEN | FOREGROUND_BLUE; // Cyan
                case Level::Info: return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; // Bright gray
                case Level::Warning: return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
                case Level::Error: return FOREGROUND_RED | FOREGROUND_INTENSITY;
                case Level::Critical: return FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY;
                default: break;
                }
                return FOREGROUND_INTENSITY; // Grey
            }());
            SetConsoleTextAttribute(console_handle, color);
            std::fwrite(formatted_line.data(), 1, formatted_line.size(), stdout);
        }
    }
    void Flush() noexcept override {}
    CONSOLE_SCREEN_BUFFER_INFO original_info = {};
    HANDLE console_handle = INVALID_HANDLE_VALUE;
    std::atomic_bool enabled = false;
#else // ^^^ Windows vvv POSIX
    explicit ColorConsoleBackend() noexcept {}
    ~ColorConsoleBackend() noexcept override {}
    void Write(const Entry& entry, std::string_view formatted_line) noexcept override {
        if (enabled.load(std::memory_order_relaxed)) {
            const std::string_view color_str = [&entry]() -> std::string_view {
                switch (entry.log_level) {
                case Level::Debug: return "\x1b[0;36m"; // Cyan
                case Level::Info: return "\x1b[0;37m"; // Bright gray
                case Level::Warning: return "\x1b[1;33m"; // Bright yellow
                case Level::Error: return "\x1b[1;31m"; // Bright red
                case Level::Critical: return "\x1b[1;35m"; // Bright magenta
                default: return "\x1b[1;30m"; // Grey
                }
            }();
            const std::string_view without_nl =
                (!formatted_line.empty() && formatted_line.back() == '\n')
                    ? formatted_line.substr(0, formatted_line.size() - 1)
                    : formatted_line;
            constexpr std::string_view reset_nl = "\x1b[0m\n";
            fmt::basic_memory_buffer<char, 2048> colored;
            colored.append(color_str.data(), color_str.data() + color_str.size());
            colored.append(without_nl.data(), without_nl.data() + without_nl.size());
            colored.append(reset_nl.data(), reset_nl.data() + reset_nl.size());
            std::fwrite(colored.data(), 1, colored.size(), stdout);
        }
    }
    void Flush() noexcept override {
        std::fflush(stdout);
    }
    std::atomic_bool enabled = false;
#endif
};

#ifndef __OPENORBIS__
/// @brief Backend that writes to a file passed into the constructor
struct FileBackend final : public Backend {
    explicit FileBackend(const std::filesystem::path filename) noexcept {
        auto old_filename = filename;
        old_filename += ".old.txt";
        // Existence checks are done within the functions themselves.
        // We don't particularly care if these succeed or not.
        void(FS::RemoveFile(old_filename));
        void(FS::RenameFile(filename, old_filename));
        file.emplace(filename, FS::FileAccessMode::Write, FS::FileType::TextFile);
    }
    ~FileBackend() noexcept override = default;

    void Write(const Entry& entry, std::string_view formatted_line) noexcept override {
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }

        std::string_view out_line = formatted_line;
#ifndef __ANDROID__
        std::string censored;
        if (Settings::values.censor_username.GetValue()) {
            // This must be a static otherwise it would get checked on EVERY
            // instance of logging an entry...
            static const std::string username = []() -> std::string {
                // in order of precedence
                // LOGNAME usually works on UNIX, USERNAME on Windows
                // Some UNIX systems suck and don't use LOGNAME so we also need USER :(
                for (auto const var : { "LOGNAME", "USERNAME", "USER", })
                    if (auto const s = ::getenv(var); s != nullptr)
                        return std::string{s};
                return std::string{};
            }();
            if (!username.empty() && out_line.find(username) != std::string_view::npos) {
                censored.assign(out_line);
                // The build version can contain the username ("Frizti v2" for user "Frizt").
                const std::string_view version{Common::g_build_version};
                const bool keep_version =
                    !version.empty() && version.find(username) != std::string_view::npos &&
                    censored.find(version) != std::string::npos;
                if (keep_version)
                    boost::replace_all(censored, version, "\x01");
                boost::replace_all(censored, username, "user");
                if (keep_version)
                    boost::replace_all(censored, "\x01", version);
                out_line = censored;
            }
        }
#endif
        const std::size_t written =
            file->WriteSpan(std::span<const char>{out_line.data(), out_line.size()});
        const std::size_t total_written =
            bytes_written.fetch_add(written, std::memory_order_relaxed) + written;

        using namespace Common::Literals;
        // Prevent logs from exceeding a set maximum size in the event that log entries are spammed.
        const auto write_limit = Settings::values.extended_logging.GetValue() ? 1_GiB : 100_MiB;
        const bool write_limit_exceeded = total_written > write_limit;
        if (write_limit_exceeded) {
            // Stop writing after the write limit is exceeded.
            // Don't close the file so we can print a stacktrace if necessary
            enabled.store(false, std::memory_order_relaxed);
        }
        if ((entry.log_level >= Level::Critical || write_limit_exceeded) &&
            !Settings::values.log_flush_line.GetValue()) {
            file->Flush();
        }
    }

    void Flush() noexcept override {
        file->Flush();
    }
private:
    std::optional<FS::IOFile> file;
    std::atomic<std::size_t> bytes_written{0};
    std::atomic_bool enabled{true};
};
#endif
/// @brief Backend that keeps the last lines in memory, off unless the debug server turns it on
struct RingBackend final : public Backend {
    void Write([[maybe_unused]] const Entry& entry,
               std::string_view formatted_line) noexcept override {
        if (capacity.load(std::memory_order_relaxed) == 0) {
            return;
        }
        try {
            const std::string_view without_nl =
                (!formatted_line.empty() && formatted_line.back() == '\n')
                    ? formatted_line.substr(0, formatted_line.size() - 1)
                    : formatted_line;
            std::string text{without_nl};
            std::scoped_lock lock{mutex};
            lines.push_back({next_sequence++, std::move(text)});
            while (lines.size() > capacity.load(std::memory_order_relaxed)) {
                lines.pop_front();
            }
        } catch (const std::exception&) {
            // Out of memory: drop the line rather than the process.
        }
    }
    void Flush() noexcept override {}

    std::atomic<std::size_t> capacity{};
    std::mutex mutex;
    std::deque<RingLine> lines;
    u64 next_sequence = 1;
};
#ifdef _WIN32
/// @brief Backend that writes to Visual Studio's output window
struct DebuggerBackend final : public Backend {
    explicit DebuggerBackend() noexcept = default;
    ~DebuggerBackend() noexcept override = default;
    void Write([[maybe_unused]] const Entry& entry,
               std::string_view formatted_line) noexcept override {
        // Without a debugger the call raises an exception internally for nothing.
        if (::IsDebuggerPresent()) {
            ::OutputDebugStringW(UTF8ToUTF16W(formatted_line).c_str());
        }
    }
    void Flush() noexcept override {}
};
#endif
#ifdef __ANDROID__
/// @brief Backend that writes to the Android logcat
struct LogcatBackend : public Backend {
    explicit LogcatBackend() noexcept = default;
    ~LogcatBackend() noexcept override = default;
    void Write(const Entry& entry,
               [[maybe_unused]] std::string_view formatted_line) noexcept override {
        android_LogPriority android_log_priority = [&]() {
            switch (entry.log_level) {
            case Level::Debug: return ANDROID_LOG_DEBUG;
            case Level::Info: return ANDROID_LOG_INFO;
            case Level::Warning: return ANDROID_LOG_WARN;
            case Level::Error: return ANDROID_LOG_ERROR;
            case Level::Critical: return ANDROID_LOG_FATAL;
            case Level::Count:
            case Level::Trace: return ANDROID_LOG_VERBOSE;
            }
        }();
        __android_log_print(android_log_priority, "YuzuNative", "%s %s:%u:%s: %.*s",
                            GetLogClassName(entry.log_class), entry.filename, entry.line_num,
                            entry.function, static_cast<int>(entry.message_len), entry.message);
    }
    void Flush() noexcept override {}
};
#endif

/// @brief Static state as a singleton.
struct Impl {
    // Well, I mean it's the default constructor!
    explicit Impl() noexcept : filter(Level::Trace) {}

    template<typename F>
    void ForEachBackend(F&& lambda) noexcept {
        lambda(static_cast<Backend&>(color_console_backend));
        lambda(static_cast<Backend&>(ring_backend));
#ifndef __OPENORBIS__
        if (file_backend)
            lambda(static_cast<Backend&>(*file_backend));
#endif
#ifdef _WIN32
        lambda(static_cast<Backend&>(debugger_backend));
#endif
#ifdef __ANDROID__
        lambda(static_cast<Backend&>(lc_backend));
#endif
    }

    Filter filter;
    ColorConsoleBackend color_console_backend{};
    RingBackend ring_backend{};
#ifndef __OPENORBIS__
    std::optional<FileBackend> file_backend;
#endif
#ifdef _WIN32
    DebuggerBackend debugger_backend{};
#endif
#ifdef __ANDROID__
    LogcatBackend lc_backend{};
#endif
    std::chrono::steady_clock::time_point time_origin{std::chrono::steady_clock::now()};
};

void PublishLowestLevel(const Filter& filter) {
    const auto lowest = std::min_element(filter.class_levels.begin(), filter.class_levels.end());
    g_lowest_enabled_level.store(u8(*lowest), std::memory_order_relaxed);
}
} // Anonymous namespace

void Filter::ParseFilterString(std::string_view filter_view) {
    auto clause_begin = filter_view.cbegin();
    while (clause_begin < filter_view.cend()) {
        auto clause_end = std::find(clause_begin, filter_view.cend(), ' ');
        // If clause isn't empty
        if (clause_end != clause_begin) {
            ParseFilterRule(*this, clause_begin, clause_end);
        }
        if (clause_end != filter_view.cend()) {
            // Skip over the whitespace
            ++clause_end;
        }
        clause_begin = clause_end;
    }
}

// Constructor shall NOT depend upon Settings() or whatever
// it's ran at global static ctor() time... so BE CAREFUL MFER!
static std::optional<Common::Log::Impl> logging_instance{};

void Initialize() {
    if (logging_instance) {
        LOG_WARNING(Log, "Reinitializing logging backend");
    } else {
        logging_instance.emplace();
        logging_instance->filter.ParseFilterString(Settings::values.log_filter.GetValue());
        PublishLowestLevel(logging_instance->filter);
#ifndef __OPENORBIS__
        using namespace Common::FS;
        const auto& log_dir = GetEdenPath(EdenPath::LogDir);
        void(CreateDir(log_dir));
        logging_instance->file_backend.emplace(log_dir / LOG_FILE);
#endif
    }
}

void Start() {
}

void Stop() {
    if (logging_instance)
        logging_instance->ForEachBackend([](Backend& backend) { backend.Flush(); });
}

void SetGlobalFilter(const Filter& filter) {
    if (logging_instance) {
        logging_instance->filter = filter;
        PublishLowestLevel(filter);
    }
}

void SetColorConsoleBackendEnabled(bool enabled) {
    if (logging_instance)
        logging_instance->color_console_backend.enabled = enabled;
}

void SetRingCapacity(std::size_t capacity) {
    if (!logging_instance)
        return;
    auto& ring = logging_instance->ring_backend;
    std::scoped_lock lock{ring.mutex};
    ring.capacity.store(capacity, std::memory_order_relaxed);
    while (ring.lines.size() > capacity) {
        ring.lines.pop_front();
    }
}

std::vector<RingLine> ReadRing(u64 after) {
    std::vector<RingLine> result;
    if (!logging_instance)
        return result;
    auto& ring = logging_instance->ring_backend;
    std::scoped_lock lock{ring.mutex};
    for (const RingLine& line : ring.lines) {
        if (line.sequence > after) {
            result.push_back(line);
        }
    }
    return result;
}

bool IsLevelEnabled(Class log_class, Level log_level) {
    return logging_instance && logging_instance->filter.CheckMessage(log_class, log_level);
}

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename, unsigned int line_num, const char* function, fmt::string_view format, const fmt::format_args& args) {
    if (!logging_instance || !logging_instance->filter.CheckMessage(log_class, log_level)) {
        return;
    }
    const bool flush = ::Settings::values.log_flush_line.GetValue();
    const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - logging_instance->time_origin);
    const char* const trimmed_filename = TrimSourcePath(filename);
    const char* const class_name = GetLogClassName(log_class);
    const char* const level_name = GetLevelName(log_level);
    const auto time_seconds = static_cast<uint32_t>(timestamp.count() / 1000000);
    const auto time_fractional = static_cast<uint32_t>(timestamp.count() % 1000000);

    fmt::basic_memory_buffer<char, 2048> line_buf;
    fmt::format_to(fmt::appender(line_buf), "[{:4d}.{:06d}] {} <{}> {}:{}:{}: ",
                   time_seconds, time_fractional, class_name, level_name,
                   trimmed_filename, line_num, function);
    const std::size_t msg_offset = line_buf.size();
    fmt::vformat_to(fmt::appender(line_buf), format, args);
    const std::size_t msg_len = line_buf.size() - msg_offset;
    line_buf.push_back('\n');

    const Entry e{
        .message = line_buf.data() + msg_offset,
        .message_len = msg_len,
        .timestamp = timestamp,
        .log_class = log_class,
        .log_level = log_level,
        .filename = trimmed_filename,
        .function = function,
        .line_num = line_num,
    };
    const std::string_view formatted_line{line_buf.data(), line_buf.size()};
    logging_instance->ForEachBackend([&](Backend& backend) {
        backend.Write(e, formatted_line);
        if (flush) {
            backend.Flush();
        }
    });
}
} // namespace Common::Log
