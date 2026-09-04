#include "session_replay.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void append_u32(std::vector<unsigned char> & bytes, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xffU));
    }
}
void append_i32(std::vector<unsigned char> & bytes, std::int32_t value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}
void append_f32(std::vector<unsigned char> & bytes, float value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

std::vector<unsigned char> fixture() {
    std::vector<unsigned char> bytes{'M', 'B', 'R', 'P', 'L', 'Y', '1', 0};
    for (const auto value : std::array<std::uint32_t, 8>{1, 30, 2, 2, 3, 1, 4, 0}) append_u32(bytes, value);
    append_i32(bytes, -1); append_i32(bytes, 0);             // parents
    append_i32(bytes, 2); append_i32(bytes, 2);              // modes
    append_i32(bytes, -1); append_i32(bytes, 0);             // frame plans
    for (float value = 0; value < 6; value += 1) append_f32(bytes, value); // qpos
    for (float value = 10; value < 22; value += 1) append_f32(bytes, value); // joints
    append_u32(bytes, 1);                                    // plan frame
    append_i32(bytes, 2);                                    // plan mode
    append_u32(bytes, 24);                                   // plan length
    for (float value = 100; value < 124; value += 1) append_f32(bytes, value); // targets
    return bytes;
}

void write(const std::filesystem::path & path, const std::vector<unsigned char> & bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("could not write test fixture");
}

bool rejected(const std::filesystem::path & path, std::vector<unsigned char> bytes) {
    write(path, bytes);
    try {
        (void)motionbricks::replay::load(path);
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "motionbricks-session-replay-test.mbreplay";
    auto bytes = fixture();
    write(path, bytes);
    const auto replay = motionbricks::replay::load(path);
    if (replay.fps != 30 || replay.frame_count != 2 || replay.joint_count != 2 ||
        replay.qpos_count != 3 || replay.plan_count != 1 || replay.frame_plans != std::vector<std::int32_t>{-1, 0} ||
        replay.frame_joints(1)[0] != 16.0F || replay.plan_targets(0)[23] != 123.0F) {
        std::cerr << "valid replay decoded incorrectly\n";
        return 1;
    }
    auto truncated = bytes;
    truncated.pop_back();
    if (!rejected(path, truncated)) {
        std::cerr << "truncated replay was accepted\n";
        return 1;
    }
    auto trailing = bytes;
    trailing.push_back(0);
    if (!rejected(path, trailing)) {
        std::cerr << "trailing data was accepted\n";
        return 1;
    }
    auto invalid_parent = bytes;
    invalid_parent[44] = 0xff; // second parent begins after the 40-byte header and root parent
    if (!rejected(path, invalid_parent)) {
        std::cerr << "invalid topology was accepted\n";
        return 1;
    }
    std::filesystem::remove(path);
    return 0;
}
