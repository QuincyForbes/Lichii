#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keeps the apps' names in one place: src/common/ui_catalog.cpp.

    python3 -I tools/ui/catalog_check.py [-v]

Fails (exit 1) when
  - a setting (src/common/settings.h, src/qt_common/config/uisettings.h) has no catalogue entry,
    an entry names a setting that doesn't exist, or an entry is duplicated;
  - a value of an enum a listed setting uses has no choice entry;
  - a {setting:...}, {choice:...}, {exp:...} or {text:...} reference in the catalogue is broken;
  - an experiment switch the iOS app or its bridge uses has no entry, or an entry no code uses;
  - an environment variable an experiment sets (EDEN_NX2_*) is missing from dist/nx2_switches.txt;
  - an app writes a catalogue name or text itself: a string literal in the iOS app's Swift files,
    the desktop app's C++ (tr(), QStringLiteral, plain literals) or .ui files that equals a setting
    or experiment name, a choice name, a catalogue text or a help text. Words too common to own
    (On, Off, Auto, ...) are allowed, and so is a literal on a line marked "catalog-check: ok" (in .ui files, a
    <string comment="catalog-check: ok">), and comment lines.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG = ROOT / "src/common/ui_catalog.cpp"

# Too common to be one entry's name: these may appear anywhere.
COMMON = {
    "On", "Off", "Auto", "Automatic", "Default", "None", "Normal", "Fast", "High", "Low", "Medium",
    "Best", "Full", "Half", "Quarter", "Never", "Always", "Ask", "Disabled", "Enabled", "Game",
    "Error", "Shop", "Japan", "USA", "Europe", "Australia", "China", "Korea", "Taiwan", "Mono",
    "Stereo", "Surround", "CPU", "GPU", "Audio", "System", "General", "Graphics", "Network",
    "Debug", "Controls", "Storage", "Users", "Advanced", "Settings", "Library", "Overlay", "Image",
    "Performance", "Emulation", "Services", "Miscellaneous", "Debugging", "Experiments", "Unsafe",
    "Safe", "Accurate", "Bilinear", "Bicubic", "Gaussian", "Lanczos", "Area", "Mitchell", "Grid",
    "List", "Console", "Docked", "Handheld", "Vulkan", "Metal", "Null", "Dynarmic", "NCE", "Global",
    "FXAA", "SMAA", "MMPX", "ScaleForce", "Zero-Tangent", "B-Spline", "Spline-1", "Boost",
    "Overclock", "Conservative", "Aggressive", "Immediate", "Balanced", "Paranoid", "Volume",
    "Language", "Region", "Mailbox", "Keyboard", "Mouse", "Vibration", "Motion", "Inserted",
    "Current Game", "Dump ExeFS", "Interface", "Hotkeys", "Upscaling", "Essentials", "Shaders",
    "Recommended", "1x", "2x", "4x", "8x", "16x", "Unlocked", "30 FPS", "60 FPS", "90 FPS", "120 FPS", "Tier",
}
# Swift/C++/ui sources the apps' texts would otherwise be written in.
SOURCES = [
    ("src/ios_frontend/App", "*.swift"),
    ("src/ios_frontend/Controls", "*.swift"),
    ("src/yuzu", "*.cpp"),
    ("src/yuzu", "*.h"),
    ("src/yuzu", "*.ui"),
    ("src/qt_common", "*.cpp"),
    ("src/qt_common", "*.h"),
]
EXCLUDED_PARTS = {"MeloController"}


def c_strings(expr):
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', expr))


def unescape(s):
    return (s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
            .replace("\\\\", "\\"))


def noop_texts(block):
    """The texts of the QT_TRANSLATE_NOOP calls in `block`, in order (None for nullptr)."""
    out = []
    for m in re.finditer(r'QT_TRANSLATE_NOOP\("\w+",\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)|nullptr', block):
        out.append(unescape(c_strings(m.group(1))) if m.group(1) else None)
    return out


def parse_catalog(problems):
    text = CATALOG.read_text(encoding="utf-8")

    def array(name):
        start = text.index(name)
        return text[start:text.index("\n};\n", start) + 1]

    settings = {}
    for m in re.finditer(r'\{"([A-Za-z0-9_]+)",(.*?)\},\n', array("const Setting kSettings[]"), re.S):
        key = m.group(1)
        if key in settings:
            problems.append(f"duplicate catalogue setting {key}")
        values = noop_texts(m.group(2))
        shown = re.search(r"\n\s+((?:Desktop|IOS|Windows|Linux|MacOS|Apple|Nowhere|Everywhere|Android)"
                          r"(?: \| \w+)*)", m.group(2))
        settings[key] = {"label": values[0] if values else None,
                         "help": values[1] if len(values) > 1 else None,
                         "reason": values[2] if len(values) > 2 else None,
                         "shown": shown.group(1) if shown else "Everywhere"}
    choices = []
    seen_choices = set()
    for m in re.finditer(r'CHOICE(?:_ON)?\((\w+), (\w+), QT_TRANSLATE_NOOP\("\w+",\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)',
                         array("const Choice kChoices[]")):
        pair = (m.group(1), m.group(2))
        if pair in seen_choices:
            problems.append(f"duplicate catalogue choice {pair[0]}.{pair[1]}")
        seen_choices.add(pair)
        choices.append((m.group(1), m.group(2), unescape(c_strings(m.group(3)))))
    texts = {}
    for m in re.finditer(r'\{"([a-z0-9_.A-Z]+)",\s*QT_TRANSLATE_NOOP\("\w+",\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)\}',
                         array("const Text kTexts[]")):
        ident = m.group(1)
        if ident in texts:
            problems.append(f"duplicate catalogue text {ident}")
        texts[ident] = unescape(c_strings(m.group(2)))
    experiments = {}
    for m in re.finditer(r'\{"(\w+)", "(\w+)",\s*QT_TRANSLATE_NOOP\("\w+",\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)\}',
                         array("const Experiment kExperiments[]")):
        ident = m.group(1)
        if ident in experiments:
            problems.append(f"duplicate catalogue experiment {ident}")
        experiments[ident] = (m.group(2), unescape(c_strings(m.group(3))))
    return settings, choices, texts, experiments


def setting_declarations():
    keys, types = {}, {}
    for f in ["src/common/settings.h", "src/qt_common/config/uisettings.h"]:
        t = (ROOT / f).read_text(encoding="utf-8")
        for m in re.finditer(r"Setting<\s*([\w:]+)[^>]*>\s*\w+\s*\{\s*linkage,(.*?)\};", t, re.S):
            lit = re.findall(r'"([A-Za-z][A-Za-z0-9_]*)"\s*,\s*Category::(\w+)', m.group(2))
            if lit:
                keys[lit[-1][0]] = lit[-1][1]
                types[lit[-1][0]] = m.group(1).split("::")[-1]
        for k, c in re.findall(r'"([A-Za-z][A-Za-z0-9_]*)"\s*,\s*Category::(\w+)', t):
            keys.setdefault(k, c)
    return keys, types


def enum_values():
    t = (ROOT / "src/common/settings_enums.h").read_text(encoding="utf-8")
    return {m.group(1): [v.strip() for v in m.group(2).split(",") if v.strip()]
            for m in re.finditer(r"^ENUM\((\w+),([^)]*)\)", t, re.M)}


def literals(path):
    """(line number, string) of each string literal written in an app source file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".ui":
        for m in re.finditer(r"<string((?: [^>]*)?)>([^<]*)</string>", text):
            if "catalog-check: ok" in m.group(1) or 'notr="true"' in m.group(1):
                continue
            yield text.count("\n", 0, m.start()) + 1, m.group(2).replace("&amp;", "&").replace(
                "&quot;", '"').replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")
        return
    lines = text.split("\n")
    for number, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if "catalog-check: ok" in line or stripped.startswith(("//", "*", "/*")):
            continue
        for m in re.finditer(r'"((?:[^"\\\n]|\\.)*)"', line):
            yield number, unescape(m.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    problems = []
    settings, choices, texts, experiments = parse_catalog(problems)

    keys, types = setting_declarations()
    for key in sorted(set(keys) - set(settings)):
        problems.append(f"setting {key} has no catalogue entry")
    for key in sorted(set(settings) - set(keys)):
        problems.append(f"catalogue entry {key} names no setting")
    enums = enum_values()
    have = {(e, v) for e, v, _ in choices}
    for key, enum in sorted(types.items()):
        entry = settings.get(key)
        if enum not in enums or entry is None or entry["label"] is None or entry["shown"] == "Nowhere":
            continue
        if enum == "AudioEngine":
            continue
        for value in enums[enum]:
            if (enum, value) not in have:
                problems.append(f"{key}: {enum}::{value} has no choice entry")

    catalog_strings = []
    for key, s in settings.items():
        for field in ("label", "help", "reason"):
            if s[field]:
                catalog_strings.append((f"setting {key} ({field})", s[field]))
    for enum, value, label in choices:
        catalog_strings.append((f"choice {enum}.{value}", label))
    for ident, text in texts.items():
        catalog_strings.append((f"text {ident}", text))
    for ident, (_, label) in experiments.items():
        catalog_strings.append((f"experiment {ident}", label))
    for where, content in catalog_strings:
        for m in re.finditer(r"\{([^{}:]+):([^{}]*)\}", content):
            kind, ref = m.group(1), m.group(2)
            if kind == "setting":
                target = settings.get(ref)
                if target is None:
                    problems.append(f"{where}: {{{kind}:{ref}}} names no setting")
                elif target["label"] is None:
                    problems.append(f"{where}: {{{kind}:{ref}}} has no label")
            elif kind == "choice":
                if "." not in ref:
                    problems.append(f"{where}: {{{kind}:{ref}}} must be key.Value")
                else:
                    skey, val = ref.split(".", 1)
                    enum = types.get(skey)
                    if skey not in keys:
                        problems.append(f"{where}: {{{kind}:{ref}}} names no setting {skey}")
                    elif enum not in enums:
                        problems.append(f"{where}: {{{kind}:{ref}}}: {skey} is not an enum setting")
                    elif (enum, val) not in have:
                        problems.append(f"{where}: {{{kind}:{ref}}}: {enum}::{val} has no choice entry")
            elif kind == "exp":
                if ref not in experiments:
                    problems.append(f"{where}: {{{kind}:{ref}}} names no experiment")
            elif kind == "text":
                if ref not in texts:
                    problems.append(f"{where}: {{{kind}:{ref}}} names no text")
            else:
                problems.append(f"{where}: unknown placeholder {{{kind}:{ref}}}")

    swift = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src/ios_frontend/App").rglob("*.swift"))
    bridge = (ROOT / "src/ios_frontend/lichii_bridge_platform.cpp").read_text(encoding="utf-8")
    used = set(re.findall(r'lichii_platform_experiment\("(\w+)"\)', bridge))
    for ident in sorted(used - set(experiments) - {"file_guest_off"}):
        problems.append(f"experiment {ident} (lichii_bridge_platform.cpp) has no catalogue entry")
    for ident in sorted(set(experiments) - used):
        problems.append(f"catalogue experiment {ident} is not used by lichii_bridge_platform.cpp")
    for ident in sorted(set(re.findall(r'experiment\("(\w+)"\)', swift)) - set(experiments)):
        problems.append(f"experiment {ident} (Swift) has no catalogue entry")
    switches = (ROOT / "dist/nx2_switches.txt").read_text(encoding="utf-8")
    for ident, block in re.findall(r'lichii_platform_experiment\("(\w+)"\)\)?\s*\{(.*?)\n    \}', bridge, re.S):
        for var in re.findall(r'"(EDEN_NX2_\w+)"', block):
            if "APPLE" not in var and var not in switches:
                problems.append(f"{var} (experiment {ident}) is missing from dist/nx2_switches.txt")
    header = (ROOT / "src/ios_frontend/lichii_bridge_platform.h").read_text(encoding="utf-8")
    for ident in experiments:
        if f'"{ident}"' in header:
            problems.append(f"lichii_bridge_platform.h describes experiment {ident}; the catalogue does")

    owned = {}
    for key, s in settings.items():
        for field in ("label", "help", "reason"):
            if s[field]:
                owned.setdefault(s[field], f"setting {key} ({field})")
    for enum, value, label in choices:
        owned.setdefault(label, f"choice {enum}.{value}")
    for ident, text in texts.items():
        owned.setdefault(text, f"text {ident}")
    for ident, (_, label) in experiments.items():
        owned.setdefault(label, f"experiment {ident}")
    owned = {k: v for k, v in owned.items() if k not in COMMON and len(k) > 2}
    exp_labels = {label: ident for ident, (_, label) in experiments.items()}

    for folder, pattern in SOURCES:
        for path in sorted((ROOT / folder).rglob(pattern)):
            if EXCLUDED_PARTS.intersection(path.parts):
                continue
            for number, literal in literals(path):
                clean = literal.strip().rstrip(":").strip()
                where = f"{path.relative_to(ROOT)}:{number}"
                if clean in owned:
                    problems.append(f"{where}: \"{clean[:60]}\" is the catalogue's {owned[clean]}")
                    continue
                for label, ident in exp_labels.items():
                    if len(label) > 12 and label in literal:
                        problems.append(f"{where}: names experiment {ident} (\"{label}\")")
                        break

    for text in list(texts.values()) + [s["help"] for s in settings.values() if s["help"]]:
        for label, ident in exp_labels.items():
            if label in text:
                problems.append(f"a catalogue text writes experiment {ident}'s name; use {{exp:{ident}}}")

    for line in problems:
        print(line)
    if args.verbose or problems:
        print(f"{len(settings)} settings, {len(choices)} choices, {len(texts)} texts, "
              f"{len(experiments)} experiments; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
