// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2017 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <climits>
#include <cerrno>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "common/logging.h"
#include "common/param_package.h"

namespace Common {

namespace {

constexpr char KEY_VALUE_SEPARATOR = ':';
constexpr char PARAM_SEPARATOR = ',';
constexpr char ESCAPE_CHARACTER = '$';

/// A placeholder for empty param packages to avoid empty strings
/// (they may be recognized as "not set" by some frontend libraries like qt)
constexpr std::string_view EMPTY_PLACEHOLDER = "[empty]";

std::string Unescape(std::string_view input) {
    if (input.find(ESCAPE_CHARACTER) == std::string_view::npos) {
        return std::string{input};
    }
    std::string out;
    out.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] == ESCAPE_CHARACTER && i + 1 < input.size()) {
            switch (input[i + 1]) {
            case '0':
                out.push_back(KEY_VALUE_SEPARATOR);
                ++i;
                continue;
            case '1':
                out.push_back(PARAM_SEPARATOR);
                ++i;
                continue;
            case '2':
                out.push_back(ESCAPE_CHARACTER);
                ++i;
                continue;
            default:
                break;
            }
        }
        out.push_back(input[i]);
    }
    return out;
}

void AppendEscaped(std::string& out, std::string_view input) {
    for (const char c : input) {
        switch (c) {
        case ESCAPE_CHARACTER:
            out.append("$2", 2);
            break;
        case PARAM_SEPARATOR:
            out.append("$1", 2);
            break;
        case KEY_VALUE_SEPARATOR:
            out.append("$0", 2);
            break;
        default:
            out.push_back(c);
            break;
        }
    }
}

const std::string* FindValue(const ParamPackage::DataType& data, const std::string& key) {
    const auto it = data.find(key);
    if (it == data.end()) {
        LOG_TRACE(Common, "key '{}' not found", key);
        return nullptr;
    }
    return &it->second;
}

} // namespace

ParamPackage::ParamPackage(const std::string& serialized) {
    if (serialized.empty() || serialized == EMPTY_PLACEHOLDER) {
        return;
    }

    std::string_view remaining{serialized};
    while (!remaining.empty()) {
        const std::size_t comma = remaining.find(PARAM_SEPARATOR);
        const std::string_view pair =
            comma == std::string_view::npos ? remaining : remaining.substr(0, comma);
        if (comma == std::string_view::npos) {
            remaining = {};
        } else {
            remaining.remove_prefix(comma + 1);
        }

        const std::size_t separator = pair.find(KEY_VALUE_SEPARATOR);
        if (separator == std::string_view::npos ||
            pair.find(KEY_VALUE_SEPARATOR, separator + 1) != std::string_view::npos) {
            LOG_ERROR(Common, "invalid key pair {}", pair);
            continue;
        }

        Set(Unescape(pair.substr(0, separator)), Unescape(pair.substr(separator + 1)));
    }
}

ParamPackage::ParamPackage(std::initializer_list<DataType::value_type> list) : data(list) {}

std::string ParamPackage::Serialize() const {
    if (data.empty()) {
        return std::string{EMPTY_PLACEHOLDER};
    }

    std::string result;
    for (const auto& [key, value] : data) {
        AppendEscaped(result, key);
        result.push_back(KEY_VALUE_SEPARATOR);
        AppendEscaped(result, value);
        result.push_back(PARAM_SEPARATOR);
    }

    result.pop_back(); // discard the trailing PARAM_SEPARATOR
    return result;
}

std::string ParamPackage::Get(const std::string& key, const std::string& default_value) const {
    const std::string* const value = FindValue(data, key);
    return value != nullptr ? *value : default_value;
}

int ParamPackage::Get(const std::string& key, int default_value) const {
    const std::string* const value = FindValue(data, key);
    if (value == nullptr) {
        return default_value;
    }

    const char* begin = value->c_str();
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(begin, &end, 10);
    if (end == begin || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX) {
        LOG_ERROR(Common, "failed to convert {} to int", *value);
        return default_value;
    }
    return static_cast<int>(parsed);
}

float ParamPackage::Get(const std::string& key, float default_value) const {
    const std::string* const value = FindValue(data, key);
    if (value == nullptr) {
        return default_value;
    }

    const char* begin = value->c_str();
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(begin, &end);
    if (end == begin || errno == ERANGE) {
        LOG_ERROR(Common, "failed to convert {} to float", *value);
        return default_value;
    }
    return parsed;
}

void ParamPackage::Set(const std::string& key, std::string value) {
    data.insert_or_assign(key, std::move(value));
}

void ParamPackage::Set(const std::string& key, int value) {
    data.insert_or_assign(key, std::to_string(value));
}

void ParamPackage::Set(const std::string& key, float value) {
    data.insert_or_assign(key, std::to_string(value));
}

bool ParamPackage::Has(const std::string& key) const {
    return data.find(key) != data.end();
}

void ParamPackage::Erase(const std::string& key) {
    data.erase(key);
}

void ParamPackage::Clear() {
    data.clear();
}

} // namespace Common
