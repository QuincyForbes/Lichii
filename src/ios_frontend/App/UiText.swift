// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import LichiiBridge

/// Swift wrapper over `common/ui_catalog` (`lichii_bridge_ui.h`), the single source of setting,
/// choice, page, overlay and experiment texts across all Lichii apps.
enum UiText {
    struct ExperimentEntry: Identifiable, Hashable {
        let id: String
        let section: String
        let label: String
    }

    static func text(_ id: String) -> String {
        id.withCString { Bridge.constant(lichii_ui_text($0)) }
    }

    /// Replaces `%1`, `%2`, ... in a catalogue text, matching Qt's `QString::arg`.
    static func format(_ id: String, _ args: String...) -> String {
        var result = text(id)
        for (offset, arg) in args.enumerated() {
            result = result.replacingOccurrences(of: "%\(offset + 1)", with: arg)
        }
        return result
    }

    static func settingLabel(_ key: String) -> String? {
        key.withCString { pointer in
            guard let raw = lichii_ui_setting_label(pointer) else { return nil }
            return String(cString: raw)
        }
    }

    static func settingHelp(_ key: String) -> String? {
        key.withCString { pointer in
            guard let raw = lichii_ui_setting_help(pointer) else { return nil }
            return String(cString: raw)
        }
    }

    static func settingReason(_ key: String) -> String? {
        key.withCString { pointer in
            guard let raw = lichii_ui_setting_reason(pointer) else { return nil }
            return String(cString: raw)
        }
    }

    static func settingShown(_ key: String) -> Bool {
        key.withCString { lichii_ui_setting_shown($0) }
    }

    static func choices(for key: String) -> [SettingChoice] {
        let count = key.withCString { lichii_ui_setting_choice_count($0) }
        guard count > 0 else { return [] }
        return (0..<count).compactMap { index in
            var value: UInt32 = 0
            var canonical: UnsafePointer<CChar>?
            var label: UnsafePointer<CChar>?
            let ok = key.withCString { pointer in
                lichii_ui_setting_choice(pointer, index, &value, &canonical, &label)
            }
            guard ok else { return nil }
            return SettingChoice(
                name: Bridge.constant(label),
                value: String(value),
                canonical: Bridge.constant(canonical)
            )
        }
    }

    static func choiceLabel(_ key: String, _ canonical: String) -> String? {
        key.withCString { k in
            canonical.withCString { c in
                guard let raw = lichii_ui_choice_label(k, c) else { return nil }
                return String(cString: raw)
            }
        }
    }

    static func experimentLabel(_ id: String) -> String {
        id.withCString { pointer in
            guard let raw = lichii_ui_experiment_label(pointer) else { return id }
            return String(cString: raw)
        }
    }

    static let experimentSections: [String] = {
        let count = lichii_ui_experiment_section_count()
        return (0..<count).map { Bridge.constant(lichii_ui_experiment_section_id($0)) }
    }()

    static let allExperiments: [ExperimentEntry] = {
        let count = lichii_ui_experiment_count()
        return (0..<count).compactMap { index in
            var id: UnsafePointer<CChar>?
            var section: UnsafePointer<CChar>?
            var label: UnsafePointer<CChar>?
            guard lichii_ui_experiment_info(index, &id, &section, &label) else { return nil }
            return ExperimentEntry(
                id: Bridge.constant(id),
                section: Bridge.constant(section),
                label: Bridge.constant(label)
            )
        }
    }()

    static func experiments(in section: String) -> [ExperimentEntry] {
        allExperiments.filter { $0.section == section }
    }
}
