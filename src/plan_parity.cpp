#include "plan_parity.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace motionbricks::parity {
namespace {

constexpr std::array<unsigned char, 8> magic{'M','B','P','A','R','I','1',0};
constexpr std::uint32_t version = 1;
constexpr std::uintmax_t maximum_bytes = 512ULL * 1024ULL * 1024ULL;

class reader {
public:
    explicit reader(const std::filesystem::path & path) : stream_(path, std::ios::binary) {
        std::error_code error;
        const auto bytes = std::filesystem::file_size(path, error);
        if (error || bytes < 36U || bytes > maximum_bytes)
            throw std::runtime_error("parity fixture size is outside supported bounds");
        if (!stream_) throw std::runtime_error("could not open parity fixture: " + path.string());
    }
    void exact(void * output, std::size_t bytes, const char * field) {
        stream_.read(static_cast<char *>(output), static_cast<std::streamsize>(bytes));
        if (stream_.gcount() != static_cast<std::streamsize>(bytes))
            throw std::runtime_error(std::string("parity fixture is truncated in ") + field);
    }
    std::uint32_t u32(const char * field) {
        std::array<unsigned char, 4> bytes{}; exact(bytes.data(), bytes.size(), field);
        return std::uint32_t(bytes[0]) | std::uint32_t(bytes[1]) << 8U |
               std::uint32_t(bytes[2]) << 16U | std::uint32_t(bytes[3]) << 24U;
    }
    std::int32_t i32(const char * field) { return std::bit_cast<std::int32_t>(u32(field)); }
    std::uint64_t u64(const char * field) {
        const auto low = u32(field), high = u32(field);
        return low | std::uint64_t(high) << 32U;
    }
    float f32(const char * field) { return std::bit_cast<float>(u32(field)); }
    void eof() {
        char value{};
        if (stream_.get(value)) throw std::runtime_error("parity fixture has trailing bytes");
        if (!stream_.eof()) throw std::runtime_error("could not finish reading parity fixture");
    }
private:
    std::ifstream stream_;
};

std::size_t count(std::initializer_list<std::uint32_t> dimensions, const char * field) {
    std::size_t result = 1;
    for (const auto dimension : dimensions) {
        if (dimension && result > std::numeric_limits<std::size_t>::max() / dimension)
            throw std::runtime_error(std::string("parity dimensions overflow in ") + field);
        result *= dimension;
    }
    return result;
}

template<class T, class Read>
std::vector<T> values(reader & input, std::size_t size, Read method, const char * field) {
    std::vector<T> result; result.reserve(size);
    for (std::size_t index = 0; index < size; ++index) result.push_back((input.*method)(field));
    return result;
}

void finite(const std::vector<float> & data, const char * field) {
    for (const auto value : data)
        if (!std::isfinite(value)) throw std::runtime_error(std::string(field) + " contains a non-finite value");
}

void quaternions(const std::vector<float> & data, const char * field) {
    if (data.size()%4U) throw std::runtime_error(std::string(field)+" has an invalid shape");
    for(std::size_t index=0;index<data.size();index+=4U) {
        const float norm=data[index]*data[index]+data[index+1]*data[index+1]+
                         data[index+2]*data[index+2]+data[index+3]*data[index+3];
        if (!(norm>0.99F && norm<1.01F))
            throw std::runtime_error(std::string(field)+" contains a non-unit quaternion");
    }
}

} // namespace

fixture load(const std::filesystem::path & path) {
    reader input(path);
    std::array<unsigned char, 8> actual{}; input.exact(actual.data(), actual.size(), "magic");
    if (actual != magic || input.u32("version") != version)
        throw std::runtime_error("unsupported parity fixture magic/version");
    fixture result;
    result.fps = input.u32("fps");
    const auto plans = input.u32("plan count");
    const auto joints = input.u32("joint count");
    const auto contexts = input.u32("context frames");
    const auto targets = input.u32("target frames");
    const auto flags = input.u32("flags");
    if (result.fps == 0 || result.fps > 1000 || plans == 0 || plans > 10000 ||
        joints != joint_count || contexts != context_frames || targets != target_frames || flags != 0)
        throw std::runtime_error("parity fixture header dimensions are unsupported");
    result.parents = values<std::int32_t>(input, joint_count, &reader::i32, "parents");
    result.neutral_joints = values<float>(input, joint_count * 3U, &reader::f32, "neutral joints");
    if (result.parents.front() != -1) throw std::runtime_error("parity root parent must be -1");
    for (std::uint32_t joint = 1; joint < joint_count; ++joint)
        if (result.parents[joint] < 0 || result.parents[joint] >= static_cast<std::int32_t>(joint))
            throw std::runtime_error("parity joint topology is invalid");
    result.plans.reserve(plans);
    for (std::uint32_t index = 0; index < plans; ++index) {
        plan item;
        item.command_frame = input.u32("command frame");
        item.expected_frames = input.u32("expected frames");
        item.mode = input.i32("mode");
        const auto reserved = input.u32("reserved");
        item.seed = input.u64("seed");
        for (auto & value : item.movement) value = input.f32("movement");
        for (auto & value : item.facing) value = input.f32("facing");
        if (item.expected_frames < 24 || item.expected_frames > 64 || item.expected_frames % 4U ||
            item.mode < 0 || item.mode >= 15 || reserved != 0)
            throw std::runtime_error("parity plan header is invalid");
        const auto frames = item.expected_frames;
        item.context_roots = values<float>(input, context_frames * 3U, &reader::f32, "context roots");
        item.context_local_rotations = values<float>(
            input, context_frames * joint_count * 4U, &reader::f32, "context rotations");
        item.expected_roots = values<float>(input, count({frames, 3U}, "expected roots"), &reader::f32, "expected roots");
        item.expected_local_rotations = values<float>(
            input, count({frames, joint_count, 4U}, "expected rotations"), &reader::f32, "expected rotations");
        item.expected_joint_positions = values<float>(
            input, count({frames, joint_count, 3U}, "expected joints"), &reader::f32, "expected joints");
        item.target_roots = values<float>(input, target_frames * 3U, &reader::f32, "target roots");
        item.target_local_rotations = values<float>(
            input, target_frames * joint_count * 4U, &reader::f32, "target rotations");
        item.target_joint_positions = values<float>(
            input, target_frames * joint_count * 3U, &reader::f32, "target joints");
        finite(item.context_roots, "context roots"); finite(item.context_local_rotations, "context rotations");
        finite(item.expected_roots, "expected roots"); finite(item.expected_local_rotations, "expected rotations");
        finite(item.expected_joint_positions, "expected joints"); finite(item.target_roots, "target roots");
        finite(item.target_local_rotations, "target rotations"); finite(item.target_joint_positions, "target joints");
        quaternions(item.context_local_rotations,"context rotations");
        quaternions(item.expected_local_rotations,"expected rotations");
        quaternions(item.target_local_rotations,"target rotations");
        result.plans.push_back(std::move(item));
    }
    finite(result.neutral_joints, "neutral joints");
    input.eof();
    return result;
}

} // namespace motionbricks::parity
