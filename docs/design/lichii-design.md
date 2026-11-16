# Lichii design language

The iOS app (`src/ios_frontend/App`, SwiftUI) is the reference. This page writes down what it does, so
the desktop app (`src/yuzu`, Qt 6) can look and read the same, and lists where each pattern lives on
the desktop. The desktop's **Lichii** theme (Settings > General > Interface > Theme, the default)
implements it; **Classic (...)** in the same list is the look before it.

## Colour

The iOS app sets one colour of its own, the accent (`Theme.accent`, red 0.55 green 0.36 blue 0.96).
Everything else is an iOS system colour, so it follows light and dark mode. The desktop copies the
values below (`src/yuzu/util/lichii_style.cpp`, `LichiiStyle::ColorsFor`) and follows the system's
light or dark mode as iOS does (Qt 6.5 and newer switch live; older Qt keeps the mode it started in).

| Token | iOS source | Light | Dark | Used for |
|---|---|---|---|---|
| accent | `Theme.accent` | `#8C5CF5` | `#8C5CF5` | tint: selected rows, switches on, prominent buttons, play badge, links |
| accent hover / pressed | (desktop only) | `#9A6EF7` / `#7B4AE6` | `#9D74F7` / `#7B4AE6` | pointer states iOS has no use for |
| accent soft | accent at 16 % / 28 % | `rgba(140,92,245,.16)` | `rgba(140,92,245,.28)` | a selected card in the grid, a found setting, a checked toolbar button |
| window | `systemGroupedBackground` (dark: elevated) | `#F2F2F7` | `#1C1C1E` | window and dialog background |
| card | `secondarySystemGroupedBackground` (dark: elevated) | `#FFFFFF` | `#2C2C2E` | a group of rows, menus, lists |
| fill | `tertiarySystemFill` over a card | `#EEEEF0` | `#3A3A3C` | text fields, pickers, buttons, tabs on hover |
| fill hover / pressed | (desktop only) | `#E5E5EA` / `#D8D8DE` | `#444446` / `#505053` | |
| text | `label` | `#000000` | `#FFFFFF` | |
| secondary | `secondaryLabel` | `#6C6C70` | `#98989F` | section headers, footers, captions, inactive tabs |
| tertiary | `tertiaryLabel` | `#AEAEB2` | `#5E5E63` | disabled text |
| separator | `opaqueSeparator` | `#C6C6C8` | `#3F3F43` | hairlines between rows |
| destructive | `systemRed` | `#FF3B30` | `#FF453A` | `role: .destructive` buttons |
| warning | `systemOrange` | `#FF9500` | `#FF9F0A` | a refused value, a slow speed in the overlay |
| success | `systemGreen` | `#34C759` | `#30D158` | |
| icon placeholder | `Theme.accent` 85 % to `.indigo` | `#8C5CF5`→`#5856D6` | same | a game without an icon (initials on the gradient) |

The dark window is iOS's *elevated* grouped background (`#1C1C1E`, what iPad sheets and Settings in
a window use), not the phone's pure black, which reads as a hole on a desktop monitor.

Overlay colours (`PerformanceOverlayView`): real FPS `#38BDF8`, displayed FPS `#4ADE80`, a speed
under 90 % orange, shaders building yellow, cards black at 55 % with a white 12 % hairline.

## Type

SwiftUI text styles, and what the desktop uses for each (the desktop keeps the system font and its
size, `F` below, as Qt sets it per platform).

| iOS style | iOS size (pt) | Weight | Where | Desktop |
|---|---|---|---|---|
| `.largeTitle` / navigation title | 34 | bold | "Library", "Settings" | page title: 1.6 F bold (`LichiiStyle::PageTitle`) |
| `.title3` | 20 | bold | "No games yet", "Recently Played" | empty state title: 1.6 F bold |
| `.headline` | 17 | semibold | game title in Game Info, settings header | tab heading on a page: 1.2 F bold |
| `.body` | 17 | regular | rows | F |
| `.subheadline` semibold | 15 | semibold | game card title | F demibold |
| section header (list) | 13 | regular, upper case | `Section("...")` headers | 0.85 F, upper case, semibold, secondary (`SectionHeader`) |
| `.footnote` | 13 | regular | section footers | 0.92 F secondary (`Footer`) |
| `.caption` | 12 | regular | setting notes, card subtitle, overlay | 0.9 F secondary (`lichiiCaption`); card subtitle 0.88 F |
| `.system(size: 11, weight: .semibold, design: .monospaced)` | 11 | semibold | performance overlay | system fixed font, 12 px, demibold |

## Spacing, size and corners

| What | iOS | Desktop |
|---|---|---|
| Group of rows (inset grouped list) | system: 10 pt radius, 16 pt side inset | `QFrame#lichiiCard` / `QGroupBox`: 10 px radius, 14 px row inset, 13 px between rows with a hairline |
| Banners (`Theme.cornerRadius`) | 14 pt | (no banners on the desktop yet) |
| Touch target | 44 pt (`.frame(minWidth: 44, minHeight: 44)`) | rows 32 px in the sidebar, toolbar buttons 30 px |
| Game icon corner | 12 % of its side (`GameIcon`) | 12 % of its side (`GameCard`) |
| Grid | `.adaptive(minimum: 150, maximum: 240)`, 16 pt between columns, 20 pt between rows | `GameGrid` columns from the icon size setting, 8 px card margin |
| Card text | title on 2 lines, subtitle 1 line, 8 pt under the icon | same |
| Play badge | 40 pt accent circle, bottom right of the icon | accent circle, a fifth of the icon, while the pointer is on the card |
| Overlay card | 10 pt radius, 8×5 pt padding, 4 pt apart | 10 px radius, 9×6 px padding, 4 px apart |
| In-game menu tile | 12 pt radius | (no in-game menu on the desktop) |
| Header icon | 44 pt, 10 pt radius | — |

## Icons

The iOS app uses SF Symbols. Apple's licence allows them only in apps for Apple platforms, so the
desktop uses [Lucide](https://lucide.dev) (ISC licence, parts MIT from Feather; `dist/qt_themes/lichii/icons/LICENSE.txt`),
whose 24-unit, 2-unit-stroke outlines match SF Symbols' regular weight closely. They ship as 64 px
masks and are tinted at runtime (`LichiiStyle::Icon`): text colour normally, white on a selected
row, tertiary when disabled.

| Place | SF Symbol (iOS) | Lucide (desktop) |
|---|---|---|
| Graphics | `display` | `monitor` |
| Controls | `gamecontroller` | `gamepad-2` |
| Audio | `speaker.wave.2` | `volume-2` |
| System | `cpu` | `cpu` |
| Advanced | `wrench.and.screwdriver` | `wrench` |
| Users | `person.2` | `users` |
| Keys & Firmware | `key` | `key-round` |
| Logs | `doc.text.magnifyingglass` | `file-search` |
| Memory & Experiments | `memorychip` | `memory-stick` |
| Recommended Settings | `wand.and.stars` | `wand-sparkles` |
| Setup Assistant | `checklist` | — |
| About This Device | `info.circle` | `info` |
| Library tab, Grid | `square.grid.2x2` | `layout-grid` |
| List | `list.bullet` | `list` |
| Settings tab | `gearshape` | `settings` |
| JIT tab | `bolt` | `zap` |
| Add Games | `plus` / `folder.badge.plus` | `folder-plus` |
| Look for New Games | `arrow.clockwise` | `refresh-cw` |
| Play | `play.fill` | drawn (filled triangle) |
| Build Shaders | `hammer` | `hammer` |
| Game Settings | `slider.horizontal.3` | `sliders-horizontal` |
| Updates, DLC & Mods | `square.stack.3d.up` | `layers` |
| Save Data | `externaldrive` | `hard-drive` (desktop: Storage page) |
| Shader Cache | `archivebox` | — |
| Delete | `trash` | `trash` |
| Search | `magnifyingglass` | `search` |
| Override per game | `globe` / `pencil.circle.fill` | the restore button of the per-game page |
| Empty library | `gamecontroller` | `gamepad-2` |
| Overlay tile | `speedometer` | `gauge` |

Desktop-only pages: General `settings`, Storage `hard-drive`.

## Layout

- **Navigation.** iOS: tabs Library, Settings, JIT; Settings is a root list that pushes one page per
  group. Desktop: the library window with a top bar (title, search, Grid, List, Add Game Folder,
  Settings), and the settings dialog with a sidebar of the same pages and a search field over it.
- **Settings root order (iOS).** Recommended Settings; Graphics, Controls, Audio, System; Games & Data
  (Keys & Firmware, Users); Shader Cache; Help (Setup Assistant, Logs, Memory & Experiments, About
  This Device); Advanced last, with "Debug switches, CPU and GPU hacks. Changing them can stop games
  from starting." **Desktop sidebar:** Graphics, Controls, Audio, System; *Games & Data*: Users,
  Storage; *App*: General (General, Interface, Web Service); Advanced (Graphics Hacks & Extensions,
  Debug) with the iOS warning as its subtitle. Recommended Settings is the dialog's button.
- **A page** is one scrolling list of groups: an upper-case header above each group, the rows on a
  rounded card, a footer under it that explains the group. One setting per row: the name left, the
  control right, an optional caption under it (iOS `SettingsCatalog.note` / the desktop tooltip text,
  shown under the row in the Lichii theme). Pages that were tabs on the desktop are shown one under
  the other with a heading each (Graphics: Graphics, Advanced Graphics). Controls keeps its tabs (one
  per player), as iOS keeps a screen per area.
- **Graphics sections** (both): Resolution & Textures, Upscaling ("The game's picture goes through the
  upscaler, then the screen filter."), Image, a section per game with its own settings (Ocarina of
  Time), Performance, Game Enhancements. The desktop adds Frame Rate (frame generation lives there on
  both, under Essentials on iOS).
- **Search** (both): matches setting names and config keys across every page; settings this
  platform has no use for are listed under "Not Available on iOS" / "Not Available on Windows" with
  the reason, as iOS's `SettingsCatalog.unavailableReasons` and the desktop's
  `ConfigurationShared::PlatformUnavailableReason`.
- **Settings that do nothing on the platform** are not shown (iOS `SettingsCatalog.availability`,
  desktop `AppliesToThisPlatform`); their values still load and save.
- **Per-game settings** (iOS "Game Settings"): the game's icon and title, the line "Overrides apply to
  this game only...", the same sections as Settings, then "Reset This Game to Recommended"
  (destructive). Desktop: the Game Settings dialog (was Properties) with Game Info on the left, the
  same note, tabs in the Settings order (Graphics, Advanced Graphics, Audio, System, CPU, Controls),
  then Updates, DLC & Mods, Network, System Applets, Graphics Hacks & Extensions, and "Reset This
  Game to Recommended...".
- **Library.** Grid of cards (icon, title on two lines, developer or format) with a menu per game, or
  a list with 60 pt icons. Empty state: `gamecontroller` symbol, "No games yet", what to add, one
  prominent button. Desktop: the same card in Grid view, the tree in List view, the same empty state
  ("Add Game Folder...").
- **In-game overlay** (Detailed preset order): card 1 Real FPS, Generated, Displayed, Frame ms, 10 %
  low; card 2 Frame gen, Scale and upscaler, Speed, Shaders building; card 3 3D, Game, Screen, GPU
  accuracy, CPU, RAM, Textures; thermal/battery icons (iOS only). The desktop overlay shows the same
  lines in the same order with VRAM after RAM and its FPS chart under the cards; it has no CPU line
  and no thermal/battery card.

## Wording

- Title Case for setting names, buttons and page titles ("Asynchronous Shader Building", "Use
  Recommended Settings"); sentence case for footers and captions; no colon after a setting name.
- Say what a setting does and its cost in one line, then the condition: "Applies the next time the
  game starts." On the desktop a setting locked while a game runs says "Stop the game to change
  this." (it is disabled there; iOS lets you change it for the next start).
- Name things by what the user sees: Graphics API (not "API"), Resolution, Upscaler, Screen Filter,
  Console Mode, Game Settings (not "Properties" or "Configure Game"), Updates, DLC & Mods (not
  "Add-Ons"), Experiments.
- Confirmations ask a question and name the action on the button: "Use the recommended settings for
  this game?" with **Reset to Recommended** (destructive) and Cancel; "Change these settings?" with
  **Use Recommended Settings**; "Delete <game>?" with **Delete**.
- Changes listed as `Name: old → new`, one per line.

`src/common/ui_catalog.h` and `src/common/ui_catalog.cpp` hold every setting label, enum choice
name, help text, unavailability reason, page and section title, performance overlay line, and
experiment switch label in one place. Qt reads it through `src/qt_common/config/catalog_text.h`
(keeping `QT_TRANSLATE_NOOP` translations), and the iOS/iPadOS app reads it through the C bridge
`src/ios_frontend/lichii_bridge_ui.h` and Swift wrapper `src/ios_frontend/App/UiText.swift`.
`tools/ui/catalog_check.py` fails (exit 1) when any setting, enum value or experiment lacks a
catalogue entry or when any Swift, C++ or `.ui` file duplicates a catalogue string.

## iPhone and iPad layout

- **iPhone** (and compact iPad Split View / Slide Over): Settings uses a `NavigationStack` with a
  gradient app-icon header (`SettingsHeader`), a horizontal `HStack`-based `JitPill`, and uniform
  29×29 rounded-square `SettingsIconBadge` (`SettingsRowLabel`) tints across every navigation row.
- **iPad** (`Theme.isPad` with `.regular` horizontal size class): `SettingsTabView` switches to a
  two-column `NavigationSplitView` (sidebar + detail `NavigationStack`, defaulting to Graphics),
  `LibraryView` widens its adaptive card grid (`185...260` pt columns) and margins, and
  `common/device_profile.cpp` identifies M1–M5 and A15–A17 Pro iPads with tablet-appropriate tiers.

## iOS pattern → Qt equivalent

| iOS (SwiftUI) | Qt (desktop) |
|---|---|
| `.tint(Theme.accent)` | `QPalette::Highlight`/`Accent`/`Link` = accent; `QPushButton:default` accent fill |
| `List` `.insetGrouped` | window colour background; `QGroupBox` or `LichiiStyle::Card` as the card |
| `Section { } header: { Text } footer: { Text }` | `LichiiStyle::SectionHeader` + `Card` + `Footer` |
| `Toggle` | `QCheckBox` drawn as a switch (`rc/switch-*.png`) |
| `Picker(...).pickerStyle(.menu)` | `QComboBox`, filled, chevron |
| `Slider` with value on the right | `QSlider` with accent fill and white knob, the value label beside it |
| `TextField` trailing | `QLineEdit`, filled |
| `Label(text, systemImage:)` row | `QListWidgetItem` with `LichiiStyle::Icon` |
| `NavigationLink` page | sidebar entry of the settings dialog |
| `.searchable(text:prompt:)` | `QLineEdit#settingsSearch` with a leading search icon over the sidebar; library search in the top bar |
| `Caption` under a setting | `QLabel#lichiiCaption` in `ConfigurationShared::Widget` |
| `.confirmationDialog` + `role: .destructive` | `QMessageBox` with a question as text, the action as a `DestructiveRole` button, Cancel default |
| `Button(...).buttonStyle(.borderedProminent)` | `QPushButton#lichiiPrimary` |
| `ContentUnavailableView.search` | "No Results for “…”" row in the settings search |
| `LazyVGrid` of `GameCard` | `GameGrid` with the `GameCard` delegate |
| `GameIcon` placeholder (initials on gradient) | the same, drawn by `GameCard` |
| `EmptyLibraryView` | `GameListPlaceholder` |
| `Menu` "Sort and View" (Grid/List) | Grid and List buttons in the library's top bar (View menu too) |
| `PerformanceOverlayView` cards | `PerformanceOverlay` cards |
| `JitPill`, thermal banners, memory banner | iOS only |
