// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Guest dialogs (error, software keyboard, profile select, controller applet) forwarded to the
// Swift UI, which shows them as UIAlertControllers and answers through
// lichii_content_dialog_respond. Answers arrive on the main thread, like the Qt frontend's.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/frontend/applets/cabinet.h"
#include "core/frontend/applets/controller.h"
#include "core/frontend/applets/error.h"
#include "core/frontend/applets/general.h"
#include "core/frontend/applets/mii_edit.h"
#include "core/frontend/applets/net_connect.h"
#include "core/frontend/applets/profile_select.h"
#include "core/frontend/applets/software_keyboard.h"
#include "core/frontend/applets/web_browser.h"
#include "core/hle/service/acc/profile_manager.h"
#include "core/hle/service/am/frontend/applets.h"
#include "ios_frontend/lichii_bridge_content.h"
#include "ios_frontend/lichii_bridge_content_host.h"

namespace LichiiContent {

namespace {

using Resolve = std::function<void(bool accepted, const std::string& text)>;

struct DialogRegistry {
    std::mutex mutex;
    std::map<u64, Resolve> pending;
    u64 next_id{};
    LichiiDialogHandler handler{};
    void* user{};
};

DialogRegistry& Registry() {
    static DialogRegistry registry;
    return registry;
}

bool HasHandler() {
    auto& registry = Registry();
    std::scoped_lock lock{registry.mutex};
    return registry.handler != nullptr;
}

// Shows a dialog; without a UI handler it resolves at once with `default_accept`.
u64 PostDialog(nlohmann::json request, Resolve resolve, bool default_accept,
               const std::string& default_text = {}) {
    auto& registry = Registry();
    std::unique_lock lock{registry.mutex};
    if (registry.handler == nullptr) {
        lock.unlock();
        LOG_INFO(Frontend, "No dialog handler, answering \"{}\" by default",
                 request.value("title", std::string{}));
        resolve(default_accept, default_text);
        return 0;
    }
    const u64 id = ++registry.next_id;
    registry.pending.emplace(id, std::move(resolve));
    const auto handler = registry.handler;
    void* const user = registry.user;
    lock.unlock();

    request["id"] = id;
    handler(user, id, request.dump().c_str());
    return id;
}

void DismissDialog(u64 id) {
    if (id == 0) {
        return;
    }
    auto& registry = Registry();
    std::unique_lock lock{registry.mutex};
    if (registry.pending.erase(id) == 0 || registry.handler == nullptr) {
        return;
    }
    const auto handler = registry.handler;
    void* const user = registry.user;
    lock.unlock();

    const nlohmann::json request{{"kind", LICHII_DIALOG_DISMISS}, {"target", id}};
    handler(user, 0, request.dump().c_str());
}

std::string ErrorCode(Result error) {
    return fmt::format("Error Code: {:04}-{:04} (0x{:08X})",
                       static_cast<u32>(error.GetModule()) + 2000, error.GetDescription(),
                       error.raw);
}

// Answers that arrive after the applet closed (or reopened) are dropped.
class Generation {
public:
    u64 Current() const {
        return value->load();
    }
    void Bump() const {
        value->fetch_add(1);
    }
    std::shared_ptr<std::atomic<u64>> Token() const {
        return value;
    }

private:
    std::shared_ptr<std::atomic<u64>> value = std::make_shared<std::atomic<u64>>(0);
};

class IosErrorApplet final : public Core::Frontend::ErrorApplet {
public:
    ~IosErrorApplet() override {
        Close();
    }

    void Close() const override {
        generation.Bump();
        DismissDialog(dialog);
        dialog = 0;
    }

    void ShowError(Result error, FinishedCallback finished) const override {
        Show(ErrorCode(error),
             "An error has occurred.\nPlease try again or contact the developer of the software.",
             std::move(finished));
    }

    void ShowErrorWithTimestamp(Result error, std::chrono::seconds time,
                                FinishedCallback finished) const override {
        const std::time_t seconds = static_cast<std::time_t>(time.count());
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &seconds);
#else
        localtime_r(&seconds, &tm);
#endif
        std::array<char, 64> when{};
        std::strftime(when.data(), when.size(), "%A, %B %e, %Y at %H:%M:%S", &tm);
        Show(ErrorCode(error),
             fmt::format("An error occurred on {}.\nPlease try again or contact the developer of "
                         "the software.",
                         when.data()),
             std::move(finished));
    }

    void ShowCustomErrorText(Result error, std::string dialog_text, std::string fullscreen_text,
                             FinishedCallback finished) const override {
        Show(ErrorCode(error),
             fmt::format("An error has occurred.\n\n{}\n\n{}", dialog_text, fullscreen_text),
             std::move(finished));
    }

private:
    void Show(std::string title, std::string message, FinishedCallback finished) const {
        LOG_WARNING(Frontend, "Guest error dialog: {}: {}", title, message);
        const auto token = generation.Token();
        const u64 expected = generation.Current();
        dialog = PostDialog(
            {{"kind", LICHII_DIALOG_ERROR},
             {"title", std::move(title)},
             {"message", std::move(message)},
             {"cancel", false}},
            [token, expected, finished = std::move(finished)](bool, const std::string&) {
                if (token->load() == expected && finished) {
                    finished();
                }
            },
            true);
    }

    Generation generation;
    mutable u64 dialog{};
};

class IosSoftwareKeyboard final : public Core::Frontend::SoftwareKeyboardApplet {
public:
    ~IosSoftwareKeyboard() override {
        Close();
    }

    void Close() const override {
        generation.Bump();
        DismissDialog(dialog);
        dialog = 0;
    }

    void InitializeKeyboard(bool is_inline,
                            Core::Frontend::KeyboardInitializeParameters initialize_parameters,
                            SubmitNormalCallback submit_normal_callback_,
                            SubmitInlineCallback submit_inline_callback_) override {
        if (is_inline) {
            submit_inline_callback = std::move(submit_inline_callback_);
        } else {
            submit_normal_callback = std::move(submit_normal_callback_);
        }
        parameters = std::move(initialize_parameters);
        current_text = parameters.initial_text;
    }

    void ShowNormalKeyboard() const override {
        const auto token = generation.Token();
        const u64 expected = generation.Current();
        auto request = KeyboardRequest(parameters.max_text_length, parameters.min_text_length,
                                       parameters.type, current_text);
        request["inline"] = false;
        request["cancel"] = !parameters.disable_cancel_button;
        request["password"] =
            parameters.password_mode == Service::AM::Frontend::SwkbdPasswordMode::Enabled;
        dialog = PostDialog(
            std::move(request),
            [this, token, expected](bool accepted, const std::string& text) {
                if (token->load() != expected || !submit_normal_callback) {
                    return;
                }
                dialog = 0;
                if (accepted) {
                    current_text = Common::UTF8ToUTF16(text);
                    submit_normal_callback(Service::AM::Frontend::SwkbdResult::Ok, current_text,
                                           false);
                } else {
                    submit_normal_callback(Service::AM::Frontend::SwkbdResult::Cancel, u"",
                                           false);
                }
            },
            true, Common::UTF16ToUTF8(current_text));
    }

    void ShowTextCheckDialog(Service::AM::Frontend::SwkbdTextCheckResult text_check_result,
                             std::u16string text_check_message) const override {
        using Service::AM::Frontend::SwkbdTextCheckResult;
        const bool confirm = text_check_result == SwkbdTextCheckResult::Confirm;
        const auto token = generation.Token();
        const u64 expected = generation.Current();
        dialog = PostDialog(
            {{"kind", LICHII_DIALOG_TEXT_CHECK},
             {"title", confirm ? "Confirm" : "Invalid text"},
             {"message", Common::UTF16ToUTF8(text_check_message)},
             {"cancel", confirm}},
            [this, token, expected, confirm](bool accepted, const std::string&) {
                if (token->load() != expected || !submit_normal_callback) {
                    return;
                }
                dialog = 0;
                if (confirm && accepted) {
                    submit_normal_callback(Service::AM::Frontend::SwkbdResult::Ok, current_text,
                                           true);
                } else if (!HasHandler()) {
                    // Nobody can type a different text; asking again would loop.
                    submit_normal_callback(Service::AM::Frontend::SwkbdResult::Cancel, u"",
                                           false);
                } else {
                    // A rejected (or not confirmed) text goes back to the keyboard.
                    ShowNormalKeyboard();
                }
            },
            true);
    }

    void ShowInlineKeyboard(Core::Frontend::InlineAppearParameters appear_parameters) const override {
        const auto token = generation.Token();
        const u64 expected = generation.Current();
        auto request = KeyboardRequest(appear_parameters.max_text_length,
                                       appear_parameters.min_text_length, appear_parameters.type,
                                       current_text);
        request["inline"] = true;
        request["cancel"] = !appear_parameters.disable_cancel_button;
        dialog = PostDialog(
            std::move(request),
            [this, token, expected](bool accepted, const std::string& text) {
                if (token->load() != expected || !submit_inline_callback) {
                    return;
                }
                dialog = 0;
                using Service::AM::Frontend::SwkbdReplyType;
                if (accepted) {
                    current_text = Common::UTF8ToUTF16(text);
                    const auto cursor = static_cast<s32>(current_text.size());
                    submit_inline_callback(SwkbdReplyType::ChangedString, current_text, cursor);
                    submit_inline_callback(SwkbdReplyType::DecidedEnter, current_text, cursor);
                } else {
                    submit_inline_callback(SwkbdReplyType::DecidedCancel, current_text,
                                           static_cast<s32>(current_text.size()));
                }
            },
            true, Common::UTF16ToUTF8(current_text));
    }

    void HideInlineKeyboard() const override {
        Close();
    }

    void InlineTextChanged(Core::Frontend::InlineTextParameters text_parameters) const override {
        current_text = std::move(text_parameters.input_text);
    }

    void ExitKeyboard() const override {
        Close();
    }

private:
    nlohmann::json KeyboardRequest(u32 max_length, u32 min_length,
                                   Service::AM::Frontend::SwkbdType type,
                                   const std::u16string& text) const {
        const auto header = Common::UTF16ToUTF8(parameters.header_text);
        return {
            {"kind", LICHII_DIALOG_KEYBOARD},
            {"title", header.empty() ? std::string{"Enter text"} : header},
            {"message", Common::UTF16ToUTF8(parameters.sub_text)},
            {"guide_text", Common::UTF16ToUTF8(parameters.guide_text)},
            {"ok_text", Common::UTF16ToUTF8(parameters.ok_text)},
            {"initial_text", Common::UTF16ToUTF8(text)},
            {"max_length", max_length},
            {"min_length", min_length},
            {"numeric", type == Service::AM::Frontend::SwkbdType::NumberPad},
        };
    }

    Core::Frontend::KeyboardInitializeParameters parameters{};
    mutable std::u16string current_text;
    mutable SubmitNormalCallback submit_normal_callback;
    mutable SubmitInlineCallback submit_inline_callback;
    Generation generation;
    mutable u64 dialog{};
};

class IosProfileSelect final : public Core::Frontend::ProfileSelectApplet {
public:
    explicit IosProfileSelect(Service::Account::ProfileManager& manager_) : manager{manager_} {}
    ~IosProfileSelect() override {
        Close();
    }

    void Close() const override {
        generation.Bump();
        DismissDialog(dialog);
        dialog = 0;
    }

    void SelectProfile(SelectProfileCallback callback,
                       const Core::Frontend::ProfileSelectParameters& parameters) const override {
        std::vector<Common::UUID> users;
        nlohmann::json options = nlohmann::json::array();
        for (const auto& uuid : manager.GetAllUsers()) {
            if (!uuid.IsValid()) {
                continue;
            }
            if (std::find(parameters.invalid_uid_list.begin(), parameters.invalid_uid_list.end(),
                          uuid) != parameters.invalid_uid_list.end()) {
                continue;
            }
            Service::Account::ProfileBase base{};
            std::string name = "User";
            if (manager.GetProfileBase(uuid, base)) {
                name = Common::StringFromFixedZeroTerminatedBuffer(
                    {reinterpret_cast<const char*>(base.username.data()), base.username.size()},
        base.username.size());
            }
            users.push_back(uuid);
            options.push_back({{"uuid", uuid.FormattedString()}, {"name", name}});
        }

        if (users.size() <= 1) {
            callback(users.empty() ? std::nullopt : std::optional{users.front()});
            return;
        }

        const auto token = generation.Token();
        const u64 expected = generation.Current();
        const auto current = static_cast<std::size_t>(
            std::max(0, Settings::values.current_user.GetValue()));
        dialog = PostDialog(
            {{"kind", LICHII_DIALOG_PROFILE},
             {"title", "Select a user"},
             {"message", ""},
             {"cancel", true},
             {"options", std::move(options)}},
            [this, token, expected, users, callback = std::move(callback)](
                bool accepted, const std::string& text) {
                if (token->load() != expected) {
                    return;
                }
                dialog = 0;
                if (!accepted) {
                    callback(std::nullopt);
                    return;
                }
                const auto index = static_cast<std::size_t>(std::strtoul(text.c_str(), nullptr, 10));
                callback(index < users.size() ? std::optional{users[index]} : std::nullopt);
            },
            true, std::to_string(std::min(current, users.size() - 1)));
    }

private:
    Service::Account::ProfileManager& manager;
    Generation generation;
    mutable u64 dialog{};
};

class IosControllerApplet final : public Core::Frontend::ControllerApplet {
public:
    explicit IosControllerApplet(Core::HID::HIDCore& hid_core) : fallback{hid_core} {}
    ~IosControllerApplet() override {
        Close();
    }

    void Close() const override {
        generation.Bump();
        DismissDialog(dialog);
        dialog = 0;
    }

    void ReconfigureControllers(ReconfigureCallback callback,
                                const Core::Frontend::ControllerParameters& parameters) const override {
        std::string message;
        if (parameters.min_players == parameters.max_players) {
            message = fmt::format("The game asks for {} player{}.", parameters.max_players,
                                  parameters.max_players == 1 ? "" : "s");
        } else {
            message = fmt::format("The game asks for {} to {} players.", parameters.min_players,
                                  parameters.max_players);
        }
        if (parameters.enable_explain_text) {
            for (const auto& text : parameters.explain_text) {
                const std::string line{text.data()};
                if (!line.empty()) {
                    message += "\n" + line;
                }
            }
        }
        message += "\nConnected controllers are assigned automatically.";

        const auto token = generation.Token();
        const u64 expected = generation.Current();
        dialog = PostDialog(
            {{"kind", LICHII_DIALOG_CONTROLLER},
             {"title", "Controllers"},
             {"message", std::move(message)},
             {"cancel", false}},
            [this, token, expected, parameters, callback = std::move(callback)](
                bool, const std::string&) {
                if (token->load() != expected) {
                    return;
                }
                dialog = 0;
                fallback.ReconfigureControllers(callback, parameters);
            },
            true);
    }

private:
    Core::Frontend::DefaultControllerApplet fallback;
    Generation generation;
    mutable u64 dialog{};
};

} // Anonymous namespace

void InstallFrontendApplets(Core::System& system) {
    system.GetFrontendAppletHolder().SetFrontendAppletSet({
        nullptr,                                                      // Amiibo Settings
        std::make_unique<IosControllerApplet>(system.HIDCore()),      // Controller Selector
        std::make_unique<IosErrorApplet>(),                           // Error Display
        nullptr,                                                      // Mii Editor
        nullptr,                                                      // Parental Controls
        nullptr,                                                      // Photo Viewer
        std::make_unique<IosProfileSelect>(system.GetProfileManager()), // Profile Selector
        std::make_unique<IosSoftwareKeyboard>(),                      // Software Keyboard
        nullptr,                                                      // Web Browser
        nullptr,                                                      // Net Connect
    });
}

} // namespace LichiiContent

extern "C" {

void lichii_content_set_dialog_handler(LichiiDialogHandler handler, void* user) {
    auto& registry = LichiiContent::Registry();
    std::unique_lock lock{registry.mutex};
    registry.handler = handler;
    registry.user = user;
}

void lichii_content_dialog_respond(uint64_t id, bool accepted, const char* text) {
    auto& registry = LichiiContent::Registry();
    LichiiContent::Resolve resolve;
    {
        std::scoped_lock lock{registry.mutex};
        const auto it = registry.pending.find(id);
        if (it == registry.pending.end()) {
            return;
        }
        resolve = std::move(it->second);
        registry.pending.erase(it);
    }
    resolve(accepted, text != nullptr ? std::string{text} : std::string{});
}

} // extern "C"
