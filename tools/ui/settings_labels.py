#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compares the setting names of the desktop app (src/qt_common/config/shared_translation.cpp) with
the iOS app's (src/ios_frontend/App/SettingsModel.swift, SettingLabels.titles).

    python3 -I tools/ui/settings_labels.py [--all]

Prints each key whose two names differ; --all prints every key both apps name. Exits 1 when a name
differs, so it can guard a change to either list. Names the iOS app derives from the key (no entry in
SettingLabels.titles) are not compared.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Desktop names that differ on purpose, with the reason (a desktop-only meaning of the same key).
ALLOWED = {
    # The desktop list offers OpenGL and Null too; iOS names the MoltenVK/Metal pair.
}


def setting_keys(text):
    """Member name -> config key, from the Setting declarations."""
    keys = {}
    for match in re.finditer(r"(\w+)\{linkage,(.*?)\};", text, re.S):
        literals = re.findall(r'"([a-z][a-z0-9_]*)"\s*,\s*(?:\n\s*)?Category::', match.group(2))
        if literals:
            keys[match.group(1)] = literals[-1]
    return keys


def unquote(expression):
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', expression)
    return "".join(parts).replace("\\'", "'").replace('\\"', '"')


def desktop_names(text, keys):
    names = {}
    for match in re.finditer(r"INSERT\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(tr\((?:[^()]|\([^()]*\))*\)|QString\(\))",
                             text, re.S):
        member = match.group(2)
        name = unquote(match.group(3)) if match.group(3).startswith("tr(") else ""
        key = keys.get(member, member)
        if name:
            names[key] = name
    return names


def ios_names(text):
    block = re.search(r"static let titles: \[String: String\] = \[(.*?)\n    \]", text, re.S)
    return dict(re.findall(r'"([a-z0-9_]+)": "((?:[^"\\]|\\.)*)"', block.group(1))) if block else {}


def setting_types(text):
    """Config key -> enum type name, from the Setting declarations."""
    types = {}
    for match in re.finditer(r"Setting<\s*(\w+)\s*(?:,\s*\w+\s*)*>\s*(\w+)\{linkage,(.*?)\};", text, re.S):
        literals = re.findall(r'"([a-z][a-z0-9_]*)"\s*,\s*(?:\n\s*)?Category::', match.group(3))
        if literals:
            types[literals[-1]] = match.group(1)
    return types


def desktop_choices(text):
    """(enum type, value name) -> desktop name."""
    return {(m.group(1), m.group(2)): unquote(m.group(3))
            for m in re.finditer(r"PAIR\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(tr\((?:[^()]|\([^()]*\))*\))", text, re.S)}


def ios_choices(text):
    block = re.search(r"static let choices: \[String: \[String: String\]\] = \[(.*?)\n    \]", text, re.S)
    choices = {}
    if block:
        for key, body in re.findall(r'"([a-z0-9_]+)": \[(.*?)\]', block.group(1), re.S):
            choices[key] = dict(re.findall(r'"(\w+)": "((?:[^"\\]|\\.)*)"', body))
    return choices


def normalise(name):
    return name.rstrip(":").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="print every key both apps name")
    args = parser.parse_args()

    settings = (ROOT / "src/common/settings.h").read_text(encoding="utf-8")
    ui_settings = (ROOT / "src/qt_common/config/uisettings.h").read_text(encoding="utf-8")
    keys = setting_keys(settings) | setting_keys(ui_settings)
    desktop = desktop_names((ROOT / "src/qt_common/config/shared_translation.cpp").read_text(encoding="utf-8"),
                            keys)
    ios = ios_names((ROOT / "src/ios_frontend/App/SettingsModel.swift").read_text(encoding="utf-8"))

    differ = 0
    for key in sorted(ios):
        if key not in desktop:
            continue
        same = normalise(desktop[key]) == ios[key] or key in ALLOWED
        if not same:
            differ += 1
        if args.all or not same:
            print(f"{'  ' if same else '! '}{key:32} iOS: {ios[key]!r:40} desktop: {desktop[key]!r}")
    print(f"{differ} of {sum(1 for key in ios if key in desktop)} names differ")

    # Choice names: reported, not failed on (a desktop list can say more, as "1X (720p/1080p)").
    swift = (ROOT / "src/ios_frontend/App/SettingsModel.swift").read_text(encoding="utf-8")
    translation = (ROOT / "src/qt_common/config/shared_translation.cpp").read_text(encoding="utf-8")
    types = setting_types(settings)
    qt_choices = desktop_choices(translation)
    choice_differ = 0
    for key, names in sorted(ios_choices(swift).items()):
        enum = types.get(key)
        for value, name in sorted(names.items()):
            qt_name = qt_choices.get((enum, value))
            if qt_name is None:
                continue
            same = qt_name == name
            choice_differ += not same
            if args.all or not same:
                print(f"{'  ' if same else '~ '}{key}.{value:18} iOS: {name!r:36} desktop: {qt_name!r}")
    print(f"{choice_differ} choice names differ")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
