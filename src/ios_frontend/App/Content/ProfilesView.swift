// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import PhotosUI
import SwiftUI
import UIKit

/// User profiles of the emulated console: the current one owns the saves games read.
struct ProfilesView: View {
    @State private var profiles: [ContentBridge.Profile] = []
    @State private var editing: ContentBridge.Profile?
    @State private var creating = false
    @State private var deleting: ContentBridge.Profile?
    @State private var alert: ContentAlert?

    var body: some View {
        List {
            Section {
                ForEach(profiles) { profile in
                    Button {
                        select(profile)
                    } label: {
                        HStack(spacing: 12) {
                            Avatar(path: profile.avatar, name: profile.name, size: 44)
                            Text(profile.name).foregroundStyle(.primary)
                            Spacer()
                            if profile.current {
                                Image(systemName: "checkmark").foregroundStyle(Theme.accent)
                            }
                        }
                    }
                    .swipeActions {
                        Button("Edit") { editing = profile }.tint(Theme.accent)
                        if profiles.count > 1 {
                            Button("Delete", role: .destructive) { deleting = profile }
                        }
                    }
                    .contextMenu {
                        Button("Edit") { editing = profile }
                        if profiles.count > 1 {
                            Button("Delete", role: .destructive) { deleting = profile }
                        }
                    }
                }
            } footer: {
                Text("Tap a user to make it current. Games save per user; games that ask for a user show a picker when there are several.")
            }
            if profiles.count < 8 {
                Section {
                    Button {
                        creating = true
                    } label: {
                        Label("Add User", systemImage: "person.badge.plus")
                    }
                }
            }
        }
        .navigationTitle("Users")
        .onAppear(perform: reload)
        .sheet(item: $editing, onDismiss: reload) { profile in
            NavigationStack { ProfileEditor(profile: profile) }
        }
        .sheet(isPresented: $creating, onDismiss: reload) {
            NavigationStack { ProfileEditor(profile: nil) }
        }
        .confirmationDialog("Delete \(deleting?.name ?? "")?",
                            isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                if let profile = deleting {
                    if !profile.uuid.withCString({ lichii_content_delete_profile($0) }) {
                        alert = ContentAlert(title: "Users", message: "This user could not be deleted.")
                    }
                }
                deleting = nil
                reload()
            }
        } message: {
            Text("The user's save files stay on disk.")
        }
        .alert(item: $alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
    }

    private func reload() {
        profiles = ContentBridge.profiles
    }

    private func select(_ profile: ContentBridge.Profile) {
        _ = profile.uuid.withCString { lichii_content_select_profile($0) }
        reload()
    }
}

private struct ProfileEditor: View {
    let profile: ContentBridge.Profile?
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var photo: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var error: String?

    var body: some View {
        Form {
            Section {
                HStack {
                    Spacer()
                    PhotosPicker(selection: $photo, matching: .images) {
                        ZStack(alignment: .bottomTrailing) {
                            if let image {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 96, height: 96)
                                    .clipShape(Circle())
                            } else {
                                Avatar(path: profile?.avatar ?? "", name: name, size: 96)
                            }
                            Image(systemName: "camera.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.white, Theme.accent)
                        }
                    }
                    Spacer()
                }
                .listRowBackground(Color.clear)
            }
            Section("Name") {
                TextField("Name", text: $name)
                    .textInputAutocapitalization(.words)
                    .autocorrectionDisabled()
                Text("Up to 32 bytes; longer names are cut.").font(.caption).foregroundStyle(.secondary)
            }
            if let error {
                Text(error).foregroundStyle(.red)
            }
        }
        .navigationTitle(profile == nil ? "New User" : "Edit User")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") { dismiss() }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button("Save", action: save)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .onAppear { name = profile?.name ?? "" }
        .onChange(of: photo) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let picked = UIImage(data: data) {
                    image = picked
                }
            }
        }
    }

    private func save() {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        var uuid = profile?.uuid
        if let uuid {
            guard uuid.withCString({ id in trimmed.withCString { lichii_content_rename_profile(id, $0) } }) else {
                error = "Renaming failed."
                return
            }
        } else {
            uuid = trimmed.withCString { ContentBridge.string(lichii_content_create_profile($0)) }
            if uuid == nil {
                error = "The user could not be created (8 users at most)."
                return
            }
        }
        if let image, let uuid, let jpeg = Self.avatarJPEG(image) {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("avatar-\(uuid).jpg")
            if (try? jpeg.write(to: url)) != nil {
                _ = uuid.withCString { id in url.path.withCString { lichii_content_set_profile_avatar(id, $0) } }
                try? FileManager.default.removeItem(at: url)
            }
        }
        dismiss()
    }

    /// Square 256x256 JPEG, the size the console's avatars use.
    static func avatarJPEG(_ image: UIImage) -> Data? {
        let side: CGFloat = 256
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format)
        let scaled = renderer.image { _ in
            let scale = max(side / image.size.width, side / image.size.height)
            let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
            image.draw(in: CGRect(x: (side - size.width) / 2, y: (side - size.height) / 2,
                                  width: size.width, height: size.height))
        }
        return scaled.jpegData(compressionQuality: 0.9)
    }
}

struct Avatar: View {
    let path: String
    let name: String
    var size: CGFloat = 44

    var body: some View {
        Group {
            if !path.isEmpty, let image = UIImage(contentsOfFile: path) {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                ZStack {
                    LinearGradient(colors: [Theme.accent, .indigo], startPoint: .topLeading,
                                   endPoint: .bottomTrailing)
                    Text(String(name.prefix(1)).uppercased())
                        .font(.system(size: size * 0.42, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
    }
}
