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
        guard let info = info(key) else { return [] }
        return (0..<info.enum_count).map { index in
            let value = key.withCString { lichii_settings_enum_value($0, index) }
            let name = Bridge.constant(key.withCString { lichii_settings_enum_name($0, index) })
            return Choice(value: String(value), name: displayName(key, name))
        }
    }

    /// "Res3_2X" -> "1.5x (1080p)" and so on; other enum names as the settings list names them.
    private static func displayName(_ key: String, _ name: String) -> String {
        guard key == "resolution_setup" else { return SettingLabels.choice(name, for: key) }
        let table: [String: String] = [
            "Res1_4X": "0.25x (180p)", "Res1_2X": "0.5x (360p)", "Res3_4X": "0.75x (540p)",
            "Res1X": "1x (720p)", "Res5_4X": "1.25x (900p)", "Res3_2X": "1.5x (1080p)",
            "Res2X": "2x (1440p)", "Res3X": "3x (2160p)", "Res4X": "4x (2880p)",
            "Res5X": "5x (3600p)", "Res6X": "6x (4320p)", "Res7X": "7x (5040p)", "Res8X": "8x (5760p)",
        ]
        return table[name] ?? name
    }
}
