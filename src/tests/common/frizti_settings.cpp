// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include "common/frizti_settings.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "frontend_common/data_manager.h"
#include "frontend_common/mod_manager.h"

namespace Frizti = Settings::Frizti;

namespace {
// What another build may leave in the shared config.
void SetOtherBuildValues(Settings::Values& values) {
    values.gpu_accuracy.SetValue(Settings::GpuAccuracy::High);
    values.dma_accuracy.SetValue(Settings::DmaAccuracy::Safe);
    values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Accurate);
    values.vsync_mode.SetValue(Settings::VSyncMode::Fifo);
    values.async_presentation.SetValue(false);
    values.use_asynchronous_shaders.SetValue(false);
    values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
    values.memory_layout_mode.SetValue(Settings::MemoryLayout::Memory_4Gb);
    values.nvdec_emulation.SetValue(Settings::NvdecEmulation::Cpu);
    values.resolution_setup.SetValue(Settings::ResolutionSetup::Res2X);
    values.cpu_accuracy.SetValue(Settings::CpuAccuracy::Accurate);
    values.use_disk_shader_cache.SetValue(false);
    values.use_vulkan_driver_pipeline_cache.SetValue(false);
#if defined(__ANDROID__) || defined(__APPLE__)
    values.dyna_state.SetValue(Settings::ExtendedDynamicState::EDS2);
#else
    values.dyna_state.SetValue(Settings::ExtendedDynamicState::Disabled);
#endif
#ifdef __ANDROID__
    values.vertex_input_dynamic_state.SetValue(true);
#else
    values.vertex_input_dynamic_state.SetValue(false);
#endif
}

void CheckFriztiValues(const Settings::Values& values) {
    CHECK(values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::Low);
    CHECK(values.dma_accuracy.GetValue() == Settings::DmaAccuracy::Default);
    CHECK(values.gpu_fence_behavior.GetValue() == Settings::GpuFenceBehavior::Default);
    CHECK(values.vsync_mode.GetValue() == Settings::VSyncMode::Mailbox);
    CHECK(values.async_presentation.GetValue());
    CHECK(values.use_asynchronous_shaders.GetValue());
    CHECK(values.use_docked_mode.GetValue() == Settings::ConsoleMode::Docked);
    CHECK(values.memory_layout_mode.GetValue() == Settings::MemoryLayout::Memory_8Gb);
    CHECK(values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu);
    CHECK(values.resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
    CHECK(values.cpu_accuracy.GetValue() == Settings::CpuAccuracy::Auto);
    CHECK(values.use_disk_shader_cache.GetValue());
    CHECK(values.use_vulkan_driver_pipeline_cache.GetValue());
#if defined(__ANDROID__) || defined(__APPLE__)
    CHECK(values.dyna_state.GetValue() == Settings::ExtendedDynamicState::Disabled);
#else
    CHECK(values.dyna_state.GetValue() == Settings::ExtendedDynamicState::EDS2);
#endif
#ifdef __ANDROID__
    CHECK_FALSE(values.vertex_input_dynamic_state.GetValue());
#else
    CHECK(values.vertex_input_dynamic_state.GetValue());
#endif
}
} // namespace

TEST_CASE("Frizti's values are the tuned settings' defaults", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    CheckFriztiValues(*values);
    for (const auto* setting : Frizti::TunedSettings(*values)) {
        CHECK(setting->Switchable());
    }
}

TEST_CASE("A new revision sets every tuned setting", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    SetOtherBuildValues(*values);
    const auto tuned = Frizti::TunedSettings(*values);
    const auto changes = Frizti::ReconcileGlobal(tuned, {}, true);
    CHECK(changes.size() == tuned.size());
    CheckFriztiValues(*values);
    CHECK(Frizti::ReconcileGlobal(tuned, {}, true).empty());
}

TEST_CASE("Between revisions only another build's changes are undone", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    const auto tuned = Frizti::TunedSettings(*values);
    // The player chose High in Frizti, and Frizti wrote it.
    values->gpu_accuracy.SetValue(Settings::GpuAccuracy::High);
    Frizti::Record record;
    record.revision = Frizti::Revision;
    record.global = Frizti::GlobalValues(tuned);

    // Another build then saved FIFO and 2x.
    values->vsync_mode.SetValue(Settings::VSyncMode::Fifo);
    values->resolution_setup.SetValue(Settings::ResolutionSetup::Res2X);
    const auto changes = Frizti::ReconcileGlobal(tuned, record, false);
    REQUIRE(changes.size() == 2);
    CHECK(values->gpu_accuracy.GetValue() == Settings::GpuAccuracy::High);
    CHECK(values->vsync_mode.GetValue() == Settings::VSyncMode::Mailbox);
    CHECK(values->resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
    CHECK(Frizti::FormatChanges(changes) ==
          "use_vsync: Fifo -> Mailbox, resolution_setup: Res2X -> Res1X");
}

TEST_CASE("Ocarina's per-game overrides go back to global", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    const auto tuned = Frizti::TunedSettings(*values);
    const auto override_value = [](auto& setting, auto value) {
        setting.SetGlobal(false);
        setting.SetValue(value);
    };
    override_value(values->gpu_accuracy, Settings::GpuAccuracy::High);
    override_value(values->vsync_mode, Settings::VSyncMode::Fifo);

    SECTION("a new revision drops every override") {
        const auto changes = Frizti::ReconcilePerGame(tuned, {}, true);
        CHECK(changes.size() == 2);
        CHECK(values->gpu_accuracy.UsingGlobal());
        CHECK(values->vsync_mode.UsingGlobal());
        CHECK(changes[0].label == "0400486024DEC000 gpu_accuracy");
        CHECK(changes[0].to == "global");
    }

    SECTION("overrides Frizti wrote are kept between revisions") {
        Frizti::Record record;
        record.per_game = Frizti::PerGameValues(tuned);
        override_value(values->memory_layout_mode, Settings::MemoryLayout::Memory_4Gb);
        const auto changes = Frizti::ReconcilePerGame(tuned, record, false);
        REQUIRE(changes.size() == 1);
        CHECK(changes[0].label == "0400486024DEC000 memory_layout_mode");
        CHECK_FALSE(values->gpu_accuracy.UsingGlobal());
        CHECK_FALSE(values->vsync_mode.UsingGlobal());
        CHECK(values->memory_layout_mode.UsingGlobal());
    }

    SECTION("per-game resolution_setup and use_docked_mode edits are always preserved") {
        // Record says both used global when frizti.ini was last written, then the user or a
        // harness edited custom/0400486024DEC000.ini directly to 0.5x Handheld.
        Frizti::Record record;
        record.revision = Frizti::Revision;
        record.per_game = Frizti::PerGameValues(tuned);

        override_value(values->resolution_setup, Settings::ResolutionSetup::Res1_2X);
        override_value(values->use_docked_mode, Settings::ConsoleMode::Handheld);

        const auto between_changes = Frizti::ReconcilePerGame(tuned, record, false);
        CHECK(between_changes.empty());
        CHECK_FALSE(values->resolution_setup.UsingGlobal());
        CHECK(values->resolution_setup.GetValue() == Settings::ResolutionSetup::Res1_2X);
        CHECK_FALSE(values->use_docked_mode.UsingGlobal());
        CHECK(values->use_docked_mode.GetValue() == Settings::ConsoleMode::Handheld);

        // Even a new revision bump drops non-preserved overrides while keeping per-game
        // resolution and docked mode intact.
        const auto full_changes = Frizti::ReconcilePerGame(tuned, {}, true);
        CHECK(full_changes.size() == size_t{2});
        CHECK(values->gpu_accuracy.UsingGlobal());
        CHECK(values->vsync_mode.UsingGlobal());
        CHECK_FALSE(values->resolution_setup.UsingGlobal());
        CHECK(values->resolution_setup.GetValue() == Settings::ResolutionSetup::Res1_2X);
        CHECK_FALSE(values->use_docked_mode.UsingGlobal());
        CHECK(values->use_docked_mode.GetValue() == Settings::ConsoleMode::Handheld);
    }

    for (const auto& restore : values->linkage.restore_functions) {
        restore();
    }
}

TEST_CASE("Docked mode replaces the handheld controller", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    values->players.GetValue(true)[0].controller_type = Settings::ControllerType::Handheld;
    CHECK(Frizti::FixDockedController(*values).size() == 1);
    CHECK(values->players.GetValue(true)[0].controller_type ==
          Settings::ControllerType::ProController);
    CHECK(Frizti::FixDockedController(*values).empty());
}

TEST_CASE("The Frizti record round-trips", "[frizti_settings]") {
    Frizti::Record record;
    record.revision = 2;
    record.global = {{"gpu_accuracy", "0"}, {"use_vsync", "1"}};
    record.per_game = {{"gpu_accuracy", "global"}, {"memory_layout_mode", "1"}};
    const auto parsed = Frizti::ParseRecord(Frizti::SerializeRecord(record));
    CHECK(parsed.revision == record.revision);
    CHECK(parsed.global == record.global);
    CHECK(parsed.per_game == record.per_game);

    const auto empty = Frizti::ParseRecord("");
    CHECK(empty.revision == 0);
    CHECK(empty.global.empty());
    CHECK(Frizti::ParseRecord("revision=x\r\nglobal.a = 1\r\n").global.at("a") == "1");
}

namespace {
class ScopedLoadAndSaveDirs {
public:
    explicit ScopedLoadAndSaveDirs(const std::u8string& subdir) {
        orig_load = Common::FS::GetEdenPath(Common::FS::EdenPath::LoadDir);
        orig_save = Common::FS::GetEdenPath(Common::FS::EdenPath::SaveDir);
        root = std::filesystem::temp_directory_path() /
               ("lichii-data-mod-test-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const auto load = root / subdir / "load";
        const auto save = root / subdir / "save";
        REQUIRE(std::filesystem::create_directories(load));
        REQUIRE(std::filesystem::create_directories(save));
        Common::FS::SetEdenPath(Common::FS::EdenPath::LoadDir, load);
        Common::FS::SetEdenPath(Common::FS::EdenPath::SaveDir, save);
    }

    ~ScopedLoadAndSaveDirs() {
        Common::FS::SetAppDirectory({});
        std::error_code ec;
        if (std::filesystem::is_directory(orig_load, ec)) {
            Common::FS::SetEdenPath(Common::FS::EdenPath::LoadDir, orig_load);
        }
        if (std::filesystem::is_directory(orig_save, ec)) {
            Common::FS::SetEdenPath(Common::FS::EdenPath::SaveDir, orig_save);
        }
        std::filesystem::remove_all(root, ec);
    }

    std::filesystem::path root;

private:
    std::filesystem::path orig_load;
    std::filesystem::path orig_save;
};
} // namespace

TEST_CASE("DataManager and ModManager preserve Unicode paths and installed mods",
          "[frontend_common][mod_manager]") {
    ScopedLoadAndSaveDirs dirs(std::u8string{u8"\u6e38\u620f-\u00e9"});
    constexpr u64 ProgramId = 0x0100000000000000ULL;

    const std::string user_id = "0123456789abcdef0123456789abcdef";
    const auto expected_save =
        Common::FS::GetEdenPath(Common::FS::EdenPath::SaveDir) / "user" / "save" /
        "0000000000000000" / user_id;
    CHECK(FrontendCommon::DataManager::GetDataDir(FrontendCommon::DataManager::DataDir::Saves,
                                                  user_id) == expected_save);
    CHECK(FrontendCommon::DataManager::GetDataDirString(FrontendCommon::DataManager::DataDir::Saves,
                                                        user_id) ==
          Common::FS::PathToUTF8String(expected_save));
    CHECK(std::filesystem::is_directory(expected_save));

    const auto mod_name = std::u8string{u8"\u6a21\u7ec4-\u00e9"};
    const auto source = dirs.root / "selected" / mod_name;
    REQUIRE(std::filesystem::create_directories(source / "romfs"));
    REQUIRE(std::ofstream(source / "romfs" / "payload", std::ios::binary).write("abc", 3).good());

    const auto installed =
        Common::FS::GetEdenPath(Common::FS::EdenPath::LoadDir) / "0100000000000000" / mod_name;
    REQUIRE(FrontendCommon::InstallModFromUTF8(Common::FS::PathToUTF8String(source), ProgramId,
                                               true) == FrontendCommon::Success);
    CHECK(std::filesystem::file_size(installed / "romfs" / "payload") == std::uintmax_t{3});

    // Reinstalling a mod directly from its installed directory (including trailing separator or
    // case difference on Windows) must not delete its own source.
    CHECK(FrontendCommon::InstallMod(installed / ".", ProgramId, false) == FrontendCommon::Success);
    CHECK(std::filesystem::file_size(installed / "romfs" / "payload") == std::uintmax_t{3});
}
