#include "session_replay.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace motionbricks::replay {
namespace {

constexpr std::array<unsigned char, 8> magic{'M', 'B', 'R', 'P', 'L', 'Y', '1', 0};
constexpr std::uint32_t version = 1;

class reader {
public:
    explicit reader(const std::filesystem::path & path) : stream_(path, std::ios::binary) {
        if (!stream_) throw std::runtime_error("could not open replay: " + path.string());
    }

    std::uint32_t u32(const char * field) {
        std::array<unsigned char, 4> bytes{};
        exact(bytes.data(), bytes.size(), field);
        return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8U) |
               (std::uint32_t(bytes[2]) << 16U) | (std::uint32_t(bytes[3]) << 24U);
    }

    std::int32_t i32(const char * field) { return std::bit_cast<std::int32_t>(u32(field)); }
    float f32(const char * field) { return std::bit_cast<float>(u32(field)); }

    void exact(void * destination, std::size_t bytes, const char * field) {
        stream_.read(static_cast<char *>(destination), static_cast<std::streamsize>(bytes));
        if (stream_.gcount() != static_cast<std::streamsize>(bytes)) {
            throw std::runtime_error(std::string("replay is truncated in ") + field);
        }
    }

    void require_eof() {
        char trailing{};
        if (stream_.get(trailing)) throw std::runtime_error("replay has trailing bytes");
        if (!stream_.eof()) throw std::runtime_error("could not finish reading replay");
    }

private:
    std::ifstream stream_;
};

std::size_t checked_count(std::initializer_list<std::uint32_t> dimensions, const char * field) {
    std::size_t result = 1;
    for (const auto dimension : dimensions) {
        if (dimension != 0 && result > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error(std::string("replay dimensions overflow in ") + field);
        }
        result *= dimension;
    }
    return result;
}

template <typename T, typename Read>
std::vector<T> read_values(reader & input, std::size_t count, Read read, const char * field) {
    std::vector<T> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) values.push_back((input.*read)(field));
    return values;
}

void finite(const std::vector<float> & values, const char * field) {
    for (const float value : values) {
        if (!std::isfinite(value)) throw std::runtime_error(std::string(field) + " contains a non-finite value");
    }
}

} // namespace

const float * session::frame_joints(std::uint32_t frame) const {
    if (frame >= frame_count) throw std::out_of_range("replay frame is out of bounds");
    return joint_positions.data() + checked_count({frame, joint_count, 3}, "frame offset");
}

const float * session::plan_targets(std::uint32_t plan) const {
    if (plan >= plan_count) throw std::out_of_range("replay plan is out of bounds");
    return target_positions.data() + checked_count({plan, target_frames, joint_count, 3}, "plan offset");
}

session load(const std::filesystem::path & path) {
    reader input(path);
    std::array<unsigned char, 8> actual_magic{};
    input.exact(actual_magic.data(), actual_magic.size(), "magic");
    if (actual_magic != magic) throw std::runtime_error("unsupported replay magic");
    if (input.u32("version") != version) throw std::runtime_error("unsupported replay version");

    session result;
    result.fps = input.u32("fps");
    result.frame_count = input.u32("frame count");
    result.joint_count = input.u32("joint count");
    result.qpos_count = input.u32("qpos count");
    result.plan_count = input.u32("plan count");
    const auto targets = input.u32("target frame count");
    const auto flags = input.u32("flags");
    if (result.fps == 0 || result.fps > 1000 || result.frame_count == 0 ||
        result.frame_count > 1'000'000 || result.joint_count == 0 || result.joint_count > 64 ||
        result.qpos_count == 0 || result.qpos_count > 256 || result.plan_count == 0 ||
        result.plan_count > result.frame_count) {
        throw std::runtime_error("replay header dimensions are outside supported bounds");
    }
    if (targets != target_frames || flags != 0) throw std::runtime_error("unsupported replay header");

    const auto payload_words = checked_count({result.joint_count}, "payload") +
        checked_count({result.frame_count, 2}, "payload") +
        checked_count({result.frame_count, result.qpos_count}, "payload") +
        checked_count({result.frame_count, result.joint_count, 3}, "payload") +
        checked_count({result.plan_count, 3}, "payload") +
        checked_count({result.plan_count, target_frames, result.joint_count, 3}, "payload");
    constexpr std::size_t max_payload_words = (512ULL * 1024ULL * 1024ULL - 40ULL) / 4ULL;
    if (payload_words > max_payload_words) throw std::runtime_error("replay payload exceeds the size limit");

    result.parents = read_values<std::int32_t>(input, result.joint_count, &reader::i32, "parents");
    result.modes = read_values<std::int32_t>(input, result.frame_count, &reader::i32, "modes");
    result.frame_plans = read_values<std::int32_t>(input, result.frame_count, &reader::i32, "frame plans");
    result.qpos = read_values<float>(input, checked_count({result.frame_count, result.qpos_count}, "qpos"), &reader::f32, "qpos");
    result.joint_positions = read_values<float>(
        input, checked_count({result.frame_count, result.joint_count, 3}, "joint positions"),
        &reader::f32, "joint positions");
    result.plan_frames = read_values<std::uint32_t>(input, result.plan_count, &reader::u32, "plan frames");
    result.plan_modes = read_values<std::int32_t>(input, result.plan_count, &reader::i32, "plan modes");
    result.plan_valid_lengths = read_values<std::uint32_t>(input, result.plan_count, &reader::u32, "plan lengths");
    result.target_positions = read_values<float>(
        input, checked_count({result.plan_count, target_frames, result.joint_count, 3}, "target positions"),
        &reader::f32, "target positions");
    input.require_eof();

    if (result.parents.front() != -1) throw std::runtime_error("replay root parent must be -1");
    for (std::uint32_t joint = 1; joint < result.joint_count; ++joint) {
        if (result.parents[joint] < 0 || result.parents[joint] >= static_cast<std::int32_t>(joint)) {
            throw std::runtime_error("replay joint topology is invalid");
        }
    }
    for (const auto plan : result.frame_plans) {
        if (plan < -1 || plan >= static_cast<std::int32_t>(result.plan_count)) {
            throw std::runtime_error("replay frame plan is out of bounds");
        }
    }
    for (std::uint32_t plan = 0; plan < result.plan_count; ++plan) {
        if (result.plan_frames[plan] >= result.frame_count ||
            (plan > 0 && result.plan_frames[plan] <= result.plan_frames[plan - 1])) {
            throw std::runtime_error("replay plan frames are invalid");
        }
    }
    finite(result.qpos, "qpos");
    finite(result.joint_positions, "joint positions");
    finite(result.target_positions, "target positions");
    return result;
}

} // namespace motionbricks::replay
