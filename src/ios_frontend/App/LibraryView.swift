// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UniformTypeIdentifiers

struct LibraryView: View {
    @EnvironmentObject private var app: AppController
    @EnvironmentObject private var library: GameLibrary
    @AppStorage("libraryGrid") private var grid = true
    @AppStorage("librarySort") private var sort: LibrarySort = .title
    @State private var query = ""
    @State private var showImporter = false
    @State private var showFolderPicker = false
    @State private var infoGame: Game?
    @State private var settingsGame: Game?
    @State private var actionSheet: (GameAction, Game)?
    @State private var deleteGame: Game?

    var body: some View {
        let games = library.sorted(sort, matching: query)
        // Only where it adds something: a small library shows every game in the grid anyway.
        let recent = query.isEmpty && sort != .recent && library.games.count > 3 ? library.recentlyPlayed : []
        ScrollView {
            if app.libraryVisible {
                VStack(alignment: .leading, spacing: 18) {
                    MemoryEntitlementBanner()
                    ThermalBanner()
                    if let progress = library.importProgress {
                        ImportProgressBanner(progress: progress)
                    }
                    if !recent.isEmpty {
                        RecentRow(games: recent, open: app.launch)
                    }
                    if games.isEmpty && !query.isEmpty && !library.games.isEmpty {
                        ContentUnavailableView.search(text: query)
                    } else if games.isEmpty {
                        EmptyLibraryView(scanning: library.isScanning, importAction: { showImporter = true })
                    } else if grid {
                        // Columns fill the width: two on a phone in portrait, more in landscape and
                        // on iPad.
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150, maximum: 240), spacing: 16,
                                                     alignment: .top)],
                                  spacing: 20) {
                            ForEach(games) { game in
                                GameCard(game: game, play: { app.launch(game) }, info: { infoGame = game }) {
                                    menu(for: game)
                                }
                                .contextMenu { menu(for: game) }
                            }
                        }
                    } else {
                        LazyVStack(spacing: 0) {
                            ForEach(games) { game in
                                HStack(spacing: 4) {
                                    Button {
                                        app.launch(game)
                                    } label: {
                                        GameRow(game: game, lastPlayed: library.playedDate(game))
                                            .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    .gameAccessibility(game) { infoGame = game }
                                    GameMenuButton(title: game.title) { menu(for: game) }
                                }
                                .contextMenu { menu(for: game) }
                                Divider().padding(.leading, 76)
                            }
                        }
                    }
                }
                .padding()
            }
        }
        .navigationTitle("Library")
        .searchable(text: $query, prompt: "Search games")
        .refreshable { await library.refresh() }
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Menu {
                    Section("Sort By") {
                        Picker("Sort By", selection: $sort) {
                            ForEach(LibrarySort.allCases) { Text($0.label).tag($0) }
                        }
                    }
                    Section("Show As") {
                        Picker("Show As", selection: $grid) {
                            Label("Grid", systemImage: "square.grid.2x2").tag(true)
                            Label("List", systemImage: "list.bullet").tag(false)
                        }
                    }
                    Section {
                        Button {
                            library.reload()
                        } label: {
                            Label("Look for New Games", systemImage: "arrow.clockwise")
                        }
                        if !library.hiddenPaths.isEmpty {
                            Button {
                                library.showHiddenGames()
                            } label: {
                                Label("Show \(library.hiddenPaths.count) Removed Game\(library.hiddenPaths.count == 1 ? "" : "s")",
                                      systemImage: "eye")
                            }
                        }
                    }
                } label: {
                    Label("Sort and View", systemImage: "line.3.horizontal.decrease.circle")
                }
                .accessibilityLabel("Sort and View")
                Menu {
                    Section("Add Games") {
                        Button {
                            showImporter = true
                        } label: {
                            Label("Copy Game Files into Lichii", systemImage: "square.and.arrow.down")
                        }
                        Button {
                            showFolderPicker = true
                        } label: {
                            Label("Link a Folder (No Copy)", systemImage: "folder.badge.plus")
                        }
                    }
                    if !library.linkedFolders.isEmpty {
                        Section("Linked Folders") {
                            ForEach(library.linkedFolders, id: \.self) { folder in
                                Button(role: .destructive) {
                                    library.unlinkFolder(folder)
                                } label: {
                                    Label("Unlink \(folder.lastPathComponent)", systemImage: "folder.badge.minus")
                                }
                            }
                        }
                    }
                } label: {
                    Label("Add Games", systemImage: "plus")
                }
                .accessibilityLabel("Add Games")
            }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: ImportTypes.games,
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                library.importItems(urls)
            }
        }
        .background(
            EmptyView()
                .fileImporter(isPresented: $showFolderPicker, allowedContentTypes: [.folder]) { result in
                    if case .success(let url) = result {
                        library.linkFolder(url)
                    }
                }
        )
        .alert(library.notice?.title ?? "", isPresented: Binding(get: { library.notice != nil },
                                                                  set: { if !$0 { library.notice = nil } }),
               presenting: library.notice) { notice in
            if !notice.removable.isEmpty {
                Button("Remove \(notice.removable.count == 1 ? "File" : "Files")", role: .destructive) {
                    library.removeImported(notice.removable)
                }
                Button("Keep", role: .cancel) {}
            } else {
                Button("OK", role: .cancel) {}
            }
        } message: { notice in
            Text(notice.message)
        }
        .confirmationDialog(deleteTitle, isPresented: Binding(get: { deleteGame != nil },
                                                              set: { if !$0 { deleteGame = nil } }),
                            titleVisibility: .visible, presenting: deleteGame) { game in
            if library.isLinked(game) {
                Button("Remove from Library", role: .destructive) { library.hide(game) }
            } else {
                Button("Delete", role: .destructive) { library.delete(game) }
            }
        } message: { game in
            Text(library.isLinked(game)
                 ? "The game stays in its linked folder in Files; Lichii stops listing it. Show Removed Games in the sort menu brings it back."
                 : "The game's file is deleted from this device.")
        }
        .sheet(item: $infoGame) { game in
            NavigationStack {
                GameInfoView(game: game, lastPlayed: library.playedDate(game))
            }
        }
        .sheet(item: $settingsGame) { game in
            NavigationStack {
                SettingsView(scope: .game(game))
            }
        }
        .sheet(isPresented: Binding(get: { actionSheet != nil }, set: { if !$0 { actionSheet = nil } })) {
            if let entry = actionSheet {
                NavigationStack {
                    entry.0.view(entry.1)
                }
            }
        }
        .onAppear {
            if library.games.isEmpty { library.reload() }
        }
    }

    private var deleteTitle: String {
        guard let game = deleteGame else { return "" }
        return library.isLinked(game) ? "Remove \(game.title) from the library?" : "Delete \(game.title)?"
    }

    @ViewBuilder
    private func menu(for game: Game) -> some View {
        Section {
            Button {
                app.launch(game)
            } label: {
                Label("Play", systemImage: "play.fill")
            }
            Button {
                infoGame = game
            } label: {
                Label("Game Info", systemImage: "info.circle")
            }
            Button {
                settingsGame = game
            } label: {
                Label("Game Settings", systemImage: "slider.horizontal.3")
            }
        }
        Section {
            ForEach(AppExtensions.gameActions) { action in
                Button {
                    actionSheet = (action, game)
                } label: {
                    Label(action.title, systemImage: action.systemImage)
                }
            }
        }
        Section {
            Button(role: .destructive) {
                deleteGame = game
            } label: {
                if library.isLinked(game) {
                    Label("Remove from Library", systemImage: "minus.circle")
                } else {
                    Label("Delete", systemImage: "trash")
                }
            }
        }
    }
}

private extension View {
    /// One VoiceOver element per game that starts it, with Game Info as an extra action.
    func gameAccessibility(_ game: Game, info: @escaping () -> Void) -> some View {
        accessibilityElement(children: .combine)
            .accessibilityLabel(game.title)
            .accessibilityHint("Starts the game")
            .accessibilityAction(named: "Game Info", info)
    }
}

/// A grid cell as wide as its column: the icon (tap to play, with a play badge), the title on up
/// to two lines and a menu button with the game's actions.
private struct GameCard<MenuContent: View>: View {
    let game: Game
    let play: () -> Void
    let info: () -> Void
    @ViewBuilder let menu: () -> MenuContent

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: play) {
                GameIcon(game: game, fills: true)
                    .shadow(color: .black.opacity(0.25), radius: 6, y: 3)
                    .overlay(alignment: .bottomTrailing) {
                        Image(systemName: "play.fill")
                            .font(.body.weight(.bold))
                            .foregroundStyle(.white)
                            .frame(width: 40, height: 40)
                            .background(Theme.accent, in: Circle())
                            .shadow(color: .black.opacity(0.3), radius: 4, y: 2)
                            .padding(8)
                    }
            }
            .buttonStyle(.plain)
            .gameAccessibility(game, info: info)
            HStack(alignment: .top, spacing: 4) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(game.title)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(2, reservesSpace: true)
                    Text(game.developer.isEmpty ? game.kind.label : game.developer)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .onTapGesture(perform: play)
                .accessibilityHidden(true)
                GameMenuButton(title: game.title, content: menu)
            }
        }
    }
}

/// "…" button with a game's actions, the same as its context menu.
private struct GameMenuButton<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        Menu(content: content) {
            Image(systemName: "ellipsis.circle")
                .font(.title3)
                .foregroundStyle(.secondary)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .accessibilityLabel("Actions for \(title)")
    }
}

private struct GameRow: View {
    let game: Game
    let lastPlayed: Date?

    var body: some View {
        HStack(spacing: 14) {
            GameIcon(game: game, size: 60)
            VStack(alignment: .leading, spacing: 3) {
                Text(game.title).font(.headline).lineLimit(1)
                Text([game.developer, game.version.isEmpty ? "" : "v\(game.version)", game.kind.label]
                    .filter { !$0.isEmpty }.joined(separator: " · "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if let lastPlayed {
                    Text("Played \(lastPlayed.formatted(.relative(presentation: .named)))")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            Spacer()
            Text(game.fileSize.byteCount)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 8)
    }
}

private struct RecentRow: View {
    let games: [Game]
    let open: @MainActor (Game) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recently Played").font(.title3.bold())
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 14) {
                    ForEach(games) { game in
                        Button {
                            open(game)
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                GameIcon(game: game, size: 96)
                                Text(game.title)
                                    .font(.caption.weight(.medium))
                                    .lineLimit(1)
                                    .frame(width: 96, alignment: .leading)
                            }
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(game.title)
                        .accessibilityHint("Starts the game")
                    }
                }
            }
        }
    }
}

private struct EmptyLibraryView: View {
    let scanning: Bool
    let importAction: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            if scanning {
                ProgressView("Looking for games...")
            } else {
                Image(systemName: "gamecontroller")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)
                Text("No games yet").font(.title3.bold())
                Text("Import .nsp, .xci, .nup, .nca or .nro files, or copy them into Lichii's folder with the Files app.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                Button("Import Games", action: importAction)
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 60)
    }
}

struct GameInfoView: View {
    @EnvironmentObject private var app: AppController
    @Environment(\.dismiss) private var dismiss
    let game: Game
    let lastPlayed: Date?

    var body: some View {
        List {
            Section {
                HStack(spacing: 16) {
                    GameIcon(game: game, size: 96)
                    VStack(alignment: .leading, spacing: 4) {
                        Text(game.title).font(.title3.bold())
                        if !game.developer.isEmpty {
                            Text(game.developer).foregroundStyle(.secondary)
                        }
                    }
                }
                Button {
                    dismiss()
                    app.launch(game)
                } label: {
                    Label("Play", systemImage: "play.fill")
                }
            }
            Section("Details") {
                LabeledContent("Title ID", value: game.programIdHex)
                LabeledContent("Version", value: game.version.isEmpty ? "-" : game.version)
                LabeledContent("Format", value: game.kind.label)
                LabeledContent("Size", value: game.fileSize.byteCount)
                if let lastPlayed {
                    LabeledContent("Last played", value: lastPlayed.formatted(date: .abbreviated, time: .shortened))
                }
                LabeledContent("File", value: game.containerURL.lastPathComponent)
            }
            Section {
                NavigationLink {
                    SettingsView(scope: .game(game))
                } label: {
                    Label("Game Settings", systemImage: "slider.horizontal.3")
                }
                ForEach(AppExtensions.gameActions) { action in
                    NavigationLink {
                        action.view(game)
                    } label: {
                        Label(action.title, systemImage: action.systemImage)
                    }
                }
            }
        }
        .navigationTitle("Game Info")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            }
        }
    }
}
