// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "common/settings.h"
#include "core/file_sys/oot_game_params.h"

namespace {

void Put32(std::vector<u8>& out, std::size_t offset, u32 value) {
    std::memcpy(out.data() + offset, &value, sizeof(value));
}

void Append32(std::vector<u8>& out, u32 value) {
    out.resize(out.size() + 4);
    Put32(out, out.size() - 4, value);
}

void AppendFloat(std::vector<u8>& out, float value) {
    u32 bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    Append32(out, bits);
}

void Align4(std::vector<u8>& out) {
    out.resize((out.size() + 3) & ~std::size_t{3});
}

// A foliage scene as the game stores it: LodSetting and LodSetting_hh share one dictionary,
// as BYML writers share equal containers.
std::vector<u8> MakeScene() {
    const std::vector<std::string_view> keys{"Arr",           "DisableTurfShdw", "Dist_LowTree",
                                             "LodSetting",    "LodSetting_hh",   "TurfDensity",
                                             "TurfMinBlades"};
    std::vector<u8> out(16);
    out[0] = 'Y';
    out[1] = 'B';
    out[2] = 7;
    const std::size_t key_table = out.size();
    out.push_back(0xC2);
    out.push_back(static_cast<u8>(keys.size()));
    out.push_back(0);
    out.push_back(0);
    const std::size_t offsets = out.size();
    out.resize(out.size() + 4 * keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        Put32(out, offsets + 4 * i, static_cast<u32>(out.size() - key_table));
        out.insert(out.end(), keys[i].begin(), keys[i].end());
        out.push_back(0);
    }
    Align4(out);

    const std::size_t lod = out.size();
    out.insert(out.end(), {0xC1, 1, 0, 0});
    out.insert(out.end(), {2, 0, 0, 0xD2});
    AppendFloat(out, 50.0f);

    const std::size_t array = out.size();
    out.insert(out.end(), {0xC0, 2, 0, 0, 0xD2, 0xD2, 0, 0});
    AppendFloat(out, 1.0f);
    AppendFloat(out, 2.0f);

    const std::size_t root = out.size();
    out.insert(out.end(), {0xC1, 6, 0, 0});
    out.insert(out.end(), {0, 0, 0, 0xC0});
    Append32(out, static_cast<u32>(array));
    out.insert(out.end(), {1, 0, 0, 0xD0});
    Append32(out, 0);
    out.insert(out.end(), {3, 0, 0, 0xC1});
    Append32(out, static_cast<u32>(lod));
    out.insert(out.end(), {4, 0, 0, 0xC1});
    Append32(out, static_cast<u32>(lod));
    out.insert(out.end(), {5, 0, 0, 0xD2});
    AppendFloat(out, 150.0f);
    out.insert(out.end(), {6, 0, 0, 0xD1});
    Append32(out, 16);

    Put32(out, 4, static_cast<u32>(key_table));
    Put32(out, 12, static_cast<u32>(root));
    return out;
}

std::vector<u8> MakeSarc(std::string_view name, const std::vector<u8>& member) {
    std::vector<u8> out{'S', 'A', 'R', 'C', 0x14, 0, 0xFF, 0xFE};
    out.resize(0x14);
    out.insert(out.end(), {'S', 'F', 'A', 'T', 0x0C, 0, 1, 0});
    Append32(out, 0x65);
    Append32(out, 0);
    Append32(out, 1u << 24);
    Append32(out, 0);
    Append32(out, static_cast<u32>(member.size()));
    out.insert(out.end(), {'S', 'F', 'N', 'T', 8, 0, 0, 0});
    out.insert(out.end(), name.begin(), name.end());
    out.push_back(0);
    out.resize((out.size() + 0xFF) & ~std::size_t{0xFF});
    Put32(out, 0xC, static_cast<u32>(out.size()));
    out.insert(out.end(), member.begin(), member.end());
    Put32(out, 8, static_cast<u32>(out.size()));
    return out;
}

constexpr std::string_view SceneName =
    "Foliage/ResFoliageScene/Test.game__foliage__ResFoliageScene.bgyml";

} // namespace

TEST_CASE("BYML editor reads and writes values in place", "[file_sys]") {
    auto scene = MakeScene();
    const auto size = scene.size();
    FileSys::BymlEditor byml{scene};
    REQUIRE(byml.IsValid());
    CHECK(byml.GetFloat({"TurfDensity"}) == 150.0f);
    CHECK(byml.GetInt({"TurfMinBlades"}) == 16);
    CHECK(byml.GetBool({"DisableTurfShdw"}) == false);
    CHECK(byml.GetFloat({"Arr", "1"}) == 2.0f);
    CHECK_FALSE(byml.GetFloat({"Arr", "2"}).has_value());
    CHECK_FALSE(byml.GetFloat({"Missing"}).has_value());
    CHECK_FALSE(byml.GetFloat({"TurfMinBlades"}).has_value());

    CHECK(byml.SetFloat({"LodSetting", "Dist_LowTree"}, 25.0f));
    // The shared dictionary is written once.
    CHECK_FALSE(byml.SetFloat({"LodSetting_hh", "Dist_LowTree"}, 12.5f));
    CHECK(byml.GetFloat({"LodSetting_hh", "Dist_LowTree"}) == 25.0f);
    CHECK_FALSE(byml.SetBool({"TurfDensity"}, true));
    CHECK(scene.size() == size);

    std::vector<u8> junk(32);
    CHECK_FALSE(FileSys::BymlEditor{junk}.IsValid());
}

TEST_CASE("Foliage levels edit the foliage scenes of a pack", "[file_sys]") {
    SECTION("Reduced") {
        auto sarc = MakeSarc(SceneName, MakeScene());
        const auto edits = FileSys::EditFoliagePack(sarc, 1);
        CHECK(edits.foliage_scenes == 1);
        CHECK(edits.foliage_values == 3);
        FileSys::ForEachSarcMember(sarc, [](std::string_view, std::span<u8> member) {
            FileSys::BymlEditor byml{member};
            CHECK(byml.GetFloat({"TurfDensity"}) == 75.0f);
            CHECK(byml.GetInt({"TurfMinBlades"}) == 8);
            CHECK(byml.GetFloat({"LodSetting_hh", "Dist_LowTree"}) == 25.0f);
            CHECK(byml.GetBool({"DisableTurfShdw"}) == false);
        });
    }
    SECTION("Low") {
        auto sarc = MakeSarc(SceneName, MakeScene());
        const auto edits = FileSys::EditFoliagePack(sarc, 2);
        CHECK(edits.foliage_values == 4);
        FileSys::ForEachSarcMember(sarc, [](std::string_view, std::span<u8> member) {
            FileSys::BymlEditor byml{member};
            CHECK(byml.GetInt({"TurfMinBlades"}) == 5);
            CHECK(byml.GetBool({"DisableTurfShdw"}) == true);
        });
    }
    SECTION("Other members and Full stay as they are") {
        const auto original = MakeSarc("Foliage/ResShootLeaf/Test.bgyml", MakeScene());
        auto sarc = original;
        CHECK(FileSys::EditFoliagePack(sarc, 2).foliage_values == 0);
        auto scene = MakeSarc(SceneName, MakeScene());
        const auto scene_original = scene;
        CHECK(FileSys::EditFoliagePack(scene, 0).foliage_values == 0);
        CHECK(sarc == original);
        CHECK(scene == scene_original);
    }
}

TEST_CASE("Ocarina graphics edits parse and apply to matching files", "[file_sys]") {
    const auto edits = FileSys::ParseOotGraphicsEdits(
        "gsys/ModelSceneConfigParam/Main:Arr.1=4.5,TurfMinBlades=3;"
        "Pack/Bootup!System/Test:DisableTurfShdw=true,broken,=1");
    REQUIRE(edits.size() == 3);
    CHECK(edits[0].file == "gsys/ModelSceneConfigParam/Main");
    CHECK(edits[0].path == std::vector<std::string>{"Arr", "1"});
    CHECK(edits[2].file == "Pack/Bootup!System/Test");
    CHECK(edits[2].value == "true");

    std::vector<FileSys::OotGraphicsEdit> mine{
        {"Test", {"Arr", "1"}, "4.5"},
        {"Test", {"TurfMinBlades"}, "3"},
        {"Test", {"DisableTurfShdw"}, "true"},
        {"Test", {"TurfDensity"}, "nope"},
        {"Other", {"TurfMinBlades"}, "9"},
    };
    auto scene = MakeScene();
    CHECK(FileSys::EditBymlFile(scene, "Test.game__foliage__ResFoliageScene.bgyml", mine) == 3);
    FileSys::BymlEditor byml{scene};
    CHECK(byml.GetFloat({"Arr", "1"}) == 4.5f);
    CHECK(byml.GetInt({"TurfMinBlades"}) == 3);
    CHECK(byml.GetBool({"DisableTurfShdw"}) == true);
    CHECK(byml.GetFloat({"TurfDensity"}) == 150.0f);
}

TEST_CASE("Ocarina graphics presets set their companions", "[file_sys]") {
    using Settings::OotGraphicsPreset;
    const auto low = Settings::OotGraphicsPresetCompanions(OotGraphicsPreset::PhoneLow);
    CHECK(low.resolution == Settings::OotResolution::Percent50);
    CHECK(low.textures == Settings::TextureQuality::Quarter);
    const auto medium = Settings::OotGraphicsPresetCompanions(OotGraphicsPreset::PhoneMedium);
    CHECK(medium.resolution == Settings::OotResolution::Percent56);
    CHECK(medium.textures == Settings::TextureQuality::Half);
    CHECK_FALSE(Settings::OotGraphicsPresetCompanions(OotGraphicsPreset::Console).resolution);
    CHECK_FALSE(Settings::OotGraphicsPresetCompanions(OotGraphicsPreset::High).textures);
    CHECK(FileSys::OotGameParamsFor(0x0100000000010000).edits.empty());
}
