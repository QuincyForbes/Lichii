// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ios_frontend/lichii_content_zip.h"

#include <algorithm>
#include <array>
#include <ctime>
#include <fstream>
#include <system_error>

#include <zlib.h>

namespace LichiiZip {

namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t LocalSig = 0x04034b50;
constexpr std::uint32_t CentralSig = 0x02014b50;
constexpr std::uint32_t EocdSig = 0x06054b50;
constexpr std::uint32_t Zip64LocatorSig = 0x07064b50;
constexpr std::uint32_t Zip64EocdSig = 0x06064b50;
constexpr std::uint32_t DescriptorSig = 0x08074b50;
constexpr std::uint32_t Max32 = 0xFFFFFFFFu;
constexpr std::size_t ChunkSize = 256 * 1024;

std::uint16_t Rd16(const unsigned char* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t Rd32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t Rd64(const unsigned char* p) {
    return static_cast<std::uint64_t>(Rd32(p)) | (static_cast<std::uint64_t>(Rd32(p + 4)) << 32);
}

void Wr16(std::string& out, std::uint32_t v) {
    out.push_back(static_cast<char>(v & 0xFF));
    out.push_back(static_cast<char>((v >> 8) & 0xFF));
}

void Wr32(std::string& out, std::uint32_t v) {
    Wr16(out, v & 0xFFFF);
    Wr16(out, v >> 16);
}

struct Entry {
    std::string name;
    std::uint16_t flags{};
    std::uint16_t method{};
    std::uint32_t crc{};
    std::uint64_t compressed_size{};
    std::uint64_t size{};
    std::uint64_t local_offset{};
};

bool ReadAt(std::ifstream& in, std::uint64_t offset, void* data, std::size_t size) {
    in.clear();
    in.seekg(static_cast<std::streamoff>(offset));
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    return in.good() || (in.eof() && static_cast<std::size_t>(in.gcount()) == size);
}

Result ReadDirectory(std::ifstream& in, std::vector<Entry>& entries) {
    in.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::uint64_t>(in.tellg());
    if (file_size < 22) {
        return Result::NotAZip;
    }

    const std::uint64_t tail_size = std::min<std::uint64_t>(file_size, 0xFFFF + 22);
    std::vector<unsigned char> tail(static_cast<std::size_t>(tail_size));
    if (!ReadAt(in, file_size - tail_size, tail.data(), tail.size())) {
        return Result::OpenFailed;
    }

    std::size_t eocd = tail.size();
    for (std::size_t i = tail.size() - 22 + 1; i-- > 0;) {
        if (Rd32(&tail[i]) == EocdSig) {
            eocd = i;
            break;
        }
    }
    if (eocd == tail.size()) {
        return Result::NotAZip;
    }

    std::uint64_t count = Rd16(&tail[eocd + 10]);
    std::uint64_t cd_size = Rd32(&tail[eocd + 12]);
    std::uint64_t cd_offset = Rd32(&tail[eocd + 16]);

    const std::uint64_t eocd_pos = file_size - tail_size + eocd;
    if (eocd_pos >= 20) {
        std::array<unsigned char, 20> locator{};
        if (ReadAt(in, eocd_pos - 20, locator.data(), locator.size()) &&
            Rd32(locator.data()) == Zip64LocatorSig) {
            std::array<unsigned char, 56> eocd64{};
            if (!ReadAt(in, Rd64(&locator[8]), eocd64.data(), eocd64.size()) ||
                Rd32(eocd64.data()) != Zip64EocdSig) {
                return Result::Corrupt;
            }
            count = Rd64(&eocd64[32]);
            cd_size = Rd64(&eocd64[40]);
            cd_offset = Rd64(&eocd64[48]);
        }
    }

    if (cd_offset + cd_size > file_size || cd_size > (256ull << 20)) {
        return Result::Corrupt;
    }

    std::vector<unsigned char> cd(static_cast<std::size_t>(cd_size));
    if (!cd.empty() && !ReadAt(in, cd_offset, cd.data(), cd.size())) {
        return Result::Corrupt;
    }

    entries.clear();
    entries.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(count, 65536)));
    std::size_t pos = 0;
    for (std::uint64_t i = 0; i < count; ++i) {
        if (pos + 46 > cd.size() || Rd32(&cd[pos]) != CentralSig) {
            return Result::Corrupt;
        }
        const unsigned char* p = &cd[pos];
        Entry entry;
        entry.flags = Rd16(p + 8);
        entry.method = Rd16(p + 10);
        entry.crc = Rd32(p + 16);
        entry.compressed_size = Rd32(p + 20);
        entry.size = Rd32(p + 24);
        const std::size_t name_len = Rd16(p + 28);
        const std::size_t extra_len = Rd16(p + 30);
        const std::size_t comment_len = Rd16(p + 32);
        entry.local_offset = Rd32(p + 42);
        if (pos + 46 + name_len + extra_len + comment_len > cd.size()) {
            return Result::Corrupt;
        }
        entry.name.assign(reinterpret_cast<const char*>(p + 46), name_len);
        std::replace(entry.name.begin(), entry.name.end(), '\\', '/');

        const unsigned char* extra = p + 46 + name_len;
        std::size_t e = 0;
        while (e + 4 <= extra_len) {
            const std::uint16_t id = Rd16(extra + e);
            const std::size_t len = Rd16(extra + e + 2);
            if (e + 4 + len > extra_len) {
                break;
            }
            if (id == 0x0001) {
                const unsigned char* field = extra + e + 4;
                std::size_t left = len;
                const auto take = [&](std::uint64_t& value) {
                    if (value == Max32 && left >= 8) {
                        value = Rd64(field);
                        field += 8;
                        left -= 8;
                    }
                };
                take(entry.size);
                take(entry.compressed_size);
                take(entry.local_offset);
            }
            e += 4 + len;
        }

        entries.push_back(std::move(entry));
        pos += 46 + name_len + extra_len + comment_len;
    }
    return Result::Success;
}

// Maps an archive name below `dest`, or returns an empty path when it would escape.
fs::path SafeTarget(const fs::path& dest, const std::string& name) {
    if (name.empty() || name.front() == '/' || name.find(':') != std::string::npos) {
        return {};
    }
    fs::path out = dest;
    std::size_t start = 0;
    while (start <= name.size()) {
        const auto slash = name.find('/', start);
        const auto part = name.substr(start, slash == std::string::npos ? std::string::npos
                                                                         : slash - start);
        if (part == "..") {
            return {};
        }
        if (!part.empty() && part != ".") {
            out /= part;
        }
        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }
    return out == dest ? fs::path{} : out;
}

Result ExtractEntry(std::ifstream& in, const Entry& entry, const fs::path& target,
                    std::uint64_t& done, std::uint64_t total, const ProgressFn& progress) {
    if ((entry.flags & 1) != 0 || (entry.method != 0 && entry.method != 8)) {
        return Result::Unsupported;
    }

    std::array<unsigned char, 30> local{};
    if (!ReadAt(in, entry.local_offset, local.data(), local.size()) ||
        Rd32(local.data()) != LocalSig) {
        return Result::Corrupt;
    }
    const std::uint64_t data_offset =
        entry.local_offset + 30 + Rd16(&local[26]) + Rd16(&local[28]);

    std::error_code ec;
    fs::create_directories(target.parent_path(), ec);
    std::ofstream out(target, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Result::WriteFailed;
    }

    in.clear();
    in.seekg(static_cast<std::streamoff>(data_offset));

    std::vector<unsigned char> input(ChunkSize);
    std::vector<unsigned char> output(ChunkSize);
    std::uint64_t remaining = entry.compressed_size;
    std::uint64_t written = 0;
    uLong crc = crc32(0, nullptr, 0);

    const auto emit = [&](const unsigned char* data, std::size_t size) {
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        crc = crc32(crc, data, static_cast<uInt>(size));
        written += size;
        done += size;
        return static_cast<bool>(out);
    };

    if (entry.method == 0) {
        while (remaining > 0) {
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, ChunkSize));
            in.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(n));
            if (static_cast<std::size_t>(in.gcount()) != n) {
                return Result::Corrupt;
            }
            remaining -= n;
            if (!emit(input.data(), n)) {
                return Result::WriteFailed;
            }
            if (progress && progress(done, total)) {
                return Result::Cancelled;
            }
        }
    } else {
        z_stream stream{};
        if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
            return Result::Corrupt;
        }
        int status = Z_OK;
        bool need_input = true;
        Result result = Result::Success;
        while (status != Z_STREAM_END) {
            if (stream.avail_in == 0 && need_input) {
                if (remaining == 0) {
                    result = Result::Corrupt;
                    break;
                }
                const auto n =
                    static_cast<std::size_t>(std::min<std::uint64_t>(remaining, ChunkSize));
                in.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(n));
                if (static_cast<std::size_t>(in.gcount()) != n) {
                    result = Result::Corrupt;
                    break;
                }
                remaining -= n;
                stream.next_in = input.data();
                stream.avail_in = static_cast<uInt>(n);
            }
            stream.next_out = output.data();
            stream.avail_out = static_cast<uInt>(output.size());
            status = inflate(&stream, Z_NO_FLUSH);
            if (status == Z_BUF_ERROR) {
                need_input = true;
                status = Z_OK;
                continue;
            }
            if (status != Z_OK && status != Z_STREAM_END) {
                result = Result::Corrupt;
                break;
            }
            need_input = stream.avail_out != 0;
            const std::size_t produced = output.size() - stream.avail_out;
            if (produced > 0 && !emit(output.data(), produced)) {
                result = Result::WriteFailed;
                break;
            }
            if (progress && progress(done, total)) {
                result = Result::Cancelled;
                break;
            }
        }
        inflateEnd(&stream);
        if (result != Result::Success) {
            return result;
        }
    }

    if (written != entry.size || static_cast<std::uint32_t>(crc) != entry.crc) {
        return Result::Corrupt;
    }
    return Result::Success;
}

void DosDateTime(std::uint16_t& date, std::uint16_t& time) {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    const int year = std::max(tm.tm_year + 1900, 1980);
    date = static_cast<std::uint16_t>(((year - 1980) << 9) | ((tm.tm_mon + 1) << 5) | tm.tm_mday);
    time = static_cast<std::uint16_t>((tm.tm_hour << 11) | (tm.tm_min << 5) | (tm.tm_sec / 2));
}

struct PendingFile {
    fs::path source;
    std::string name;
    bool directory{};
};

struct CentralRecord {
    std::string name;
    std::uint16_t flags{};
    std::uint16_t method{};
    std::uint32_t crc{};
    std::uint32_t compressed_size{};
    std::uint32_t size{};
    std::uint32_t offset{};
    bool directory{};
};

std::string JoinName(const std::string& prefix, const std::string& rest) {
    if (prefix.empty()) {
        return rest;
    }
    if (rest.empty()) {
        return prefix;
    }
    return prefix.back() == '/' ? prefix + rest : prefix + "/" + rest;
}

std::vector<PendingFile> Collect(const std::vector<Input>& inputs, std::uint64_t& total) {
    std::vector<PendingFile> files;
    std::error_code ec;
    for (const auto& input : inputs) {
        if (fs::is_directory(input.source, ec)) {
            if (!input.name.empty()) {
                files.push_back({input.source, JoinName(input.name, ""), true});
            }
            std::vector<fs::path> children;
            for (auto it = fs::recursive_directory_iterator(
                     input.source, fs::directory_options::skip_permission_denied, ec);
                 !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
                children.push_back(it->path());
            }
            std::sort(children.begin(), children.end());
            for (const auto& child : children) {
                const auto relative = child.lexically_relative(input.source).generic_u8string();
                const std::string rel(relative.begin(), relative.end());
                const bool is_dir = fs::is_directory(child, ec);
                if (!is_dir) {
                    total += fs::file_size(child, ec);
                }
                files.push_back({child, JoinName(input.name, rel), is_dir});
            }
        } else if (fs::is_regular_file(input.source, ec)) {
            total += fs::file_size(input.source, ec);
            files.push_back({input.source, input.name, false});
        }
    }
    return files;
}

} // Anonymous namespace

const char* ResultString(Result result) {
    switch (result) {
    case Result::Success:
        return "Success";
    case Result::OpenFailed:
        return "The archive could not be opened";
    case Result::NotAZip:
        return "The file is not a ZIP archive";
    case Result::Unsupported:
        return "The archive uses encryption or an unsupported compression method";
    case Result::Corrupt:
        return "The archive is damaged";
    case Result::WriteFailed:
        return "Writing the files failed (is the storage full?)";
    case Result::TooLarge:
        return "The data is too large for a ZIP archive";
    case Result::Cancelled:
        return "Cancelled";
    }
    return "Unknown error";
}

std::vector<std::string> List(const fs::path& zip) {
    std::ifstream in(zip, std::ios::binary);
    std::vector<Entry> entries;
    if (!in || ReadDirectory(in, entries) != Result::Success) {
        return {};
    }
    std::vector<std::string> names;
    names.reserve(entries.size());
    for (auto& entry : entries) {
        names.push_back(std::move(entry.name));
    }
    return names;
}

Result Extract(const fs::path& zip, const fs::path& dest, const ProgressFn& progress) {
    std::ifstream in(zip, std::ios::binary);
    if (!in) {
        return Result::OpenFailed;
    }
    std::vector<Entry> entries;
    if (const auto result = ReadDirectory(in, entries); result != Result::Success) {
        return result;
    }

    std::error_code ec;
    fs::create_directories(dest, ec);

    std::uint64_t total = 0;
    for (const auto& entry : entries) {
        total += entry.size;
    }
    std::uint64_t done = 0;

    for (const auto& entry : entries) {
        const auto target = SafeTarget(dest, entry.name);
        if (target.empty()) {
            continue;
        }
        if (entry.name.back() == '/') {
            fs::create_directories(target, ec);
            continue;
        }
        if (const auto result = ExtractEntry(in, entry, target, done, total, progress);
            result != Result::Success) {
            return result;
        }
    }
    if (progress) {
        progress(total, total);
    }
    return Result::Success;
}

Result Create(const fs::path& zip, const std::vector<Input>& inputs, const ProgressFn& progress) {
    std::uint64_t total = 0;
    const auto files = Collect(inputs, total);
    if (files.size() > 0xFFFF) {
        return Result::TooLarge;
    }

    std::error_code ec;
    if (zip.has_parent_path()) {
        fs::create_directories(zip.parent_path(), ec);
    }
    std::ofstream out(zip, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Result::WriteFailed;
    }

    std::uint16_t dos_date{};
    std::uint16_t dos_time{};
    DosDateTime(dos_date, dos_time);

    std::vector<CentralRecord> records;
    records.reserve(files.size());
    std::vector<unsigned char> input(ChunkSize);
    std::vector<unsigned char> output(ChunkSize);
    std::uint64_t offset = 0;
    std::uint64_t done = 0;

    const auto put = [&](const std::string& bytes) {
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        offset += bytes.size();
        return static_cast<bool>(out);
    };

    for (const auto& file : files) {
        if (offset > Max32) {
            return Result::TooLarge;
        }
        CentralRecord record;
        record.name = file.directory ? JoinName(file.name, "") + "/" : file.name;
        if (file.directory && record.name.size() >= 2 &&
            record.name[record.name.size() - 2] == '/') {
            record.name.pop_back();
        }
        record.directory = file.directory;
        record.flags = file.directory ? 0x0800 : 0x0808;
        record.method = file.directory ? 0 : 8;
        record.offset = static_cast<std::uint32_t>(offset);

        std::string header;
        Wr32(header, LocalSig);
        Wr16(header, 20);
        Wr16(header, record.flags);
        Wr16(header, record.method);
        Wr16(header, dos_time);
        Wr16(header, dos_date);
        Wr32(header, 0);
        Wr32(header, 0);
        Wr32(header, 0);
        Wr16(header, static_cast<std::uint32_t>(record.name.size()));
        Wr16(header, 0);
        header += record.name;
        if (!put(header)) {
            return Result::WriteFailed;
        }

        if (!file.directory) {
            std::ifstream in(file.source, std::ios::binary);
            if (!in) {
                return Result::OpenFailed;
            }
            z_stream stream{};
            if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                             Z_DEFAULT_STRATEGY) != Z_OK) {
                return Result::WriteFailed;
            }
            uLong crc = crc32(0, nullptr, 0);
            std::uint64_t size = 0;
            std::uint64_t compressed = 0;
            Result result = Result::Success;
            bool finished = false;
            while (!finished) {
                in.read(reinterpret_cast<char*>(input.data()),
                        static_cast<std::streamsize>(input.size()));
                const auto n = static_cast<std::size_t>(in.gcount());
                const bool last = n < input.size();
                crc = crc32(crc, input.data(), static_cast<uInt>(n));
                size += n;
                done += n;
                stream.next_in = input.data();
                stream.avail_in = static_cast<uInt>(n);
                do {
                    stream.next_out = output.data();
                    stream.avail_out = static_cast<uInt>(output.size());
                    const int status = deflate(&stream, last ? Z_FINISH : Z_NO_FLUSH);
                    if (status == Z_STREAM_ERROR) {
                        result = Result::WriteFailed;
                        break;
                    }
                    const std::size_t produced = output.size() - stream.avail_out;
                    out.write(reinterpret_cast<const char*>(output.data()),
                              static_cast<std::streamsize>(produced));
                    compressed += produced;
                    if (!out) {
                        result = Result::WriteFailed;
                        break;
                    }
                    if (status == Z_STREAM_END) {
                        finished = true;
                    }
                } while (stream.avail_out == 0 && !finished);
                if (result != Result::Success) {
                    break;
                }
                if (progress && progress(done, total)) {
                    result = Result::Cancelled;
                    break;
                }
            }
            deflateEnd(&stream);
            if (result != Result::Success) {
                return result;
            }
            offset += compressed;
            if (size > Max32 || compressed > Max32) {
                return Result::TooLarge;
            }
            record.crc = static_cast<std::uint32_t>(crc);
            record.size = static_cast<std::uint32_t>(size);
            record.compressed_size = static_cast<std::uint32_t>(compressed);

            std::string descriptor;
            Wr32(descriptor, DescriptorSig);
            Wr32(descriptor, record.crc);
            Wr32(descriptor, record.compressed_size);
            Wr32(descriptor, record.size);
            if (!put(descriptor)) {
                return Result::WriteFailed;
            }
        }
        records.push_back(std::move(record));
    }

    if (offset > Max32) {
        return Result::TooLarge;
    }
    const std::uint64_t cd_offset = offset;
    for (const auto& record : records) {
        std::string central;
        Wr32(central, CentralSig);
        Wr16(central, 20);
        Wr16(central, 20);
        Wr16(central, record.flags);
        Wr16(central, record.method);
        Wr16(central, dos_time);
        Wr16(central, dos_date);
        Wr32(central, record.crc);
        Wr32(central, record.compressed_size);
        Wr32(central, record.size);
        Wr16(central, static_cast<std::uint32_t>(record.name.size()));
        Wr16(central, 0);
        Wr16(central, 0);
        Wr16(central, 0);
        Wr16(central, 0);
        Wr32(central, record.directory ? 0x10 : 0);
        Wr32(central, record.offset);
        central += record.name;
        if (!put(central)) {
            return Result::WriteFailed;
        }
    }
    const std::uint64_t cd_size = offset - cd_offset;
    if (offset > Max32) {
        return Result::TooLarge;
    }

    std::string eocd;
    Wr32(eocd, EocdSig);
    Wr16(eocd, 0);
    Wr16(eocd, 0);
    Wr16(eocd, static_cast<std::uint32_t>(records.size()));
    Wr16(eocd, static_cast<std::uint32_t>(records.size()));
    Wr32(eocd, static_cast<std::uint32_t>(cd_size));
    Wr32(eocd, static_cast<std::uint32_t>(cd_offset));
    Wr16(eocd, 0);
    if (!put(eocd)) {
        return Result::WriteFailed;
    }
    out.close();
    if (progress) {
        progress(total, total);
    }
    return out ? Result::Success : Result::WriteFailed;
}

} // namespace LichiiZip
