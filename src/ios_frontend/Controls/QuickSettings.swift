// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge

/// The settings the in-game menu changes, through the settings bridge: the value in effect, which
/// is the running game's own where its per-game config overrides the setting.
@MainActor
enum QuickSettings {
    struct Choice: Identifiable, Hashable {
        let value: String
        let name: String
        var id: String { value }
    }

    static func string(_ key: String) -> String {
        Bridge.string(key.withCString { lichii_settings_get_effective($0) })
    }

    static func bool(_ key: String) -> Bool {
        string(key) == "true"
    }

    static func int(_ key: String) -> Int {
        Int(string(key)) ?? 0
    }

    /// An enum setting's value in effect by its enum name ("MetalFx"), "" when unknown.
    static func canonical(_ key: String) -> String {
        guard let info = info(key) else { return "" }
        let value = string(key)
        for index in 0..<info.enum_count where
            String(key.withCString { lichii_settings_enum_value($0, index) }) == value {
            return Bridge.constant(key.withCString { lichii_settings_enum_name($0, index) })
        }
        return ""
    }

    /// Sets and saves to the per-game or the global config, wherever the value comes from; a
    /// running game picks up runtime-modifiable values at once.
    static func set(_ key: String, _ value: String) {
        _ = key.withCString { k in value.withCString { v in lichii_settings_set_effective(k, v) } }
    }

    static func set(_ key: String, _ value: Bool) {
        set(key, value ? "true" : "false")
    }

    static func set(_ key: String, _ value: Int) {
        set(key, String(value))
    }

    static func info(_ key: String) -> LichiiSettingInfo? {
        var info = LichiiSettingInfo()
        return key.withCString { lichii_settings_find($0, &info) } ? info : nil
    }

    /// False when the change waits for the next boot.
    static func appliesLive(_ key: String) -> Bool {
        info(key)?.runtime_modifiable ?? false
    }

    static func range(_ key: String) -> ClosedRange<Int>? {
        guard let info = info(key), info.ranged else { return nil }
        return Int(info.minimum)...Int(info.maximum)
    }

    static func choices(_ key: String) -> [Choice] {
        let catalogChoices = UiText.choices(for: key)
        if !catalogChoices.isEmpty {
            return catalogChoices.map { Choice(value: $0.value, name: $0.name) }
        }
        guard let info = info(key) else { return [] }
        return (0..<info.enum_count).map { index in
            let value = key.withCString { lichii_settings_enum_value($0, index) }
            let name = Bridge.constant(key.withCString { lichii_settings_enum_name($0, index) })
            return Choice(value: String(value), name: SettingLabels.choice(name, for: key))
        }
    }
}
