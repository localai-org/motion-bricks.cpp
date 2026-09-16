#include <motionbricks/motionbricks.h>

#include "agent.hpp"
#include "handles.hpp"
#include "planner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

template <typename T, void (*Free)(T *)>
using handle = std::unique_ptr<T, decltype(Free)>;

struct statistics {
    double mean{};
    double p50{};
    double p95{};
    double minimum{};
    double maximum{};
};

statistics summarize(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("cannot summarize an empty sample");
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double q) {
        const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(values.size())));
        return values[std::max<std::size_t>(1U, rank) - 1U];
    };
    return {std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size()),
            percentile(0.50), percentile(0.95), values.front(), values.back()};
}

void print_statistics(const statistics & value) {
    std::cout << "{\"mean_ms\":" << value.mean
              << ",\"p50_ms\":" << value.p50
              << ",\"p95_ms\":" << value.p95
              << ",\"min_ms\":" << value.minimum
              << ",\"max_ms\":" << value.maximum << '}';
}

std::uint32_t parse_positive(const char * text, const char * name) {
    char * end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value == 0U || value > UINT32_MAX)
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    return static_cast<std::uint32_t>(value);
}

void check(mb_status status, std::string_view operation, const char * error) {
    if (status != MB_OK)
        throw std::runtime_error(std::string(operation) + ": " + error);
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 7 && argc != 8) {
        std::cerr << "usage: motionbricks-cpu-profile BUNDLE STYLE THREADS REQUESTS WARMUP ROUNDS [TOKENS]\n";
        return 2;
    }
    const auto threads = parse_positive(argv[3], "threads");
    const auto requests = parse_positive(argv[4], "requests");
    const auto warmup = parse_positive(argv[5], "warmup");
    const auto rounds = parse_positive(argv[6], "rounds");
    if (requests > 64U) throw std::runtime_error("requests must not exceed 64");

    std::array<char, 1024> error{};
    mb_runtime_options * raw_options = nullptr;
    check(mb_runtime_options_create(&raw_options, error.data(), error.size()),
          "create runtime options", error.data());
    handle<mb_runtime_options, mb_runtime_options_free> options(raw_options, mb_runtime_options_free);
    check(mb_runtime_options_set_device(options.get(), MB_DEVICE_CPU, error.data(), error.size()),
          "select CPU", error.data());
    check(mb_runtime_options_set_threads(options.get(), threads, error.data(), error.size()),
          "set threads", error.data());

    mb_model * raw_model = nullptr;
    check(mb_model_load(argv[1], options.get(), &raw_model, error.data(), error.size()),
          "load model", error.data());
    handle<mb_model, mb_model_free> model(raw_model, mb_model_free);
    mb_style * raw_style = nullptr;
    check(mb_style_load(model.get(), argv[2], &raw_style, error.data(), error.size()),
          "load style", error.data());
    handle<mb_style, mb_style_free> style(raw_style, mb_style_free);
    std::uint32_t forced_tokens = 0U;
    if (argc == 8) {
        forced_tokens = parse_positive(argv[7], "tokens");
        if (forced_tokens < 6U || forced_tokens > 16U)
            throw std::runtime_error("tokens must be in the range 6..16");
        raw_style->allowed_tokens.fill(0U);
        raw_style->allowed_tokens[forced_tokens - 6U] = 1U;
    }

    std::vector<handle<mb_agent, mb_agent_free>> agents;
    std::vector<handle<mb_command, mb_command_free>> commands;
    for (std::uint32_t i = 0; i < requests; ++i) {
        mb_agent * raw_agent = nullptr;
        mb_command * raw_command = nullptr;
        check(mb_agent_create(model.get(), &raw_agent, error.data(), error.size()),
              "create agent", error.data());
        agents.emplace_back(raw_agent, mb_agent_free);
        check(mb_agent_reset(agents.back().get(), style.get(), error.data(), error.size()),
              "reset agent", error.data());
        check(mb_command_create(&raw_command, error.data(), error.size()),
              "create command", error.data());
        commands.emplace_back(raw_command, mb_command_free);
        check(mb_command_set_style(commands.back().get(), style.get(), error.data(), error.size()),
              "set style", error.data());
        check(mb_command_set_movement_direction(commands.back().get(), 0.0F, 0.0F, 1.0F,
                                                error.data(), error.size()),
              "set movement", error.data());
        check(mb_command_set_facing_direction(commands.back().get(), 0.0F, 0.0F, 1.0F,
                                              error.data(), error.size()),
              "set facing", error.data());
        check(mb_command_set_seed(commands.back().get(), UINT64_C(1009) + i,
                                  error.data(), error.size()),
              "set seed", error.data());
    }

    const auto run_one = [&](std::uint32_t request, std::uint64_t * frames) {
        mb_motion * raw_motion = nullptr;
        const auto begin = clock_type::now();
        check(mb_agent_plan(agents[request].get(), commands[request].get(), &raw_motion,
                            error.data(), error.size()), "plan", error.data());
        const auto end = clock_type::now();
        handle<mb_motion, mb_motion_free> motion(raw_motion, mb_motion_free);
        check(mb_motion_get_frame_count(motion.get(), frames, error.data(), error.size()),
              "get frame count", error.data());
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::vector<std::uint64_t> frame_counts(requests);
    for (std::uint32_t round = 0; round < warmup; ++round)
        for (std::uint32_t request = 0; request < requests; ++request)
            (void)run_one(request, &frame_counts[request]);

    std::vector<double> request_ms;
    std::vector<double> round_ms;
    request_ms.reserve(static_cast<std::size_t>(rounds) * requests);
    round_ms.reserve(rounds);
    for (std::uint32_t round = 0; round < rounds; ++round) {
        const auto begin = clock_type::now();
        for (std::uint32_t request = 0; request < requests; ++request)
            request_ms.push_back(run_one(request, &frame_counts[request]));
        const auto end = clock_type::now();
        round_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }

    constexpr std::uint32_t trace_rounds = 3U;
    std::array<std::vector<double>, 4> stage_samples;
    std::vector<std::uint32_t> selected_tokens;
    for (std::uint32_t trace_round = 0; trace_round < trace_rounds; ++trace_round) {
        for (std::uint32_t request = 0; request < requests; ++request) {
            motionbricks::detail::transition_trace trace;
            mb_motion motion;
            std::string reason;
            const auto status = motionbricks::detail::plan_agent_trace(
                *agents[request], *commands[request], motion, trace, reason);
            if (status != MB_OK) throw std::runtime_error("trace plan: " + reason);
            for (std::size_t stage = 0; stage < stage_samples.size(); ++stage)
                stage_samples[stage].push_back(trace.stage_ms[stage]);
            selected_tokens.push_back(trace.selected_tokens);
        }
    }

    std::cout << std::fixed << std::setprecision(3)
              << "{\"threads\":" << threads
              << ",\"requests\":" << requests
              << ",\"forced_tokens\":" << forced_tokens
              << ",\"warmup_rounds\":" << warmup
              << ",\"measured_rounds\":" << rounds
              << ",\"request_latency\":";
    print_statistics(summarize(request_ms));
    std::cout << ",\"round_latency\":";
    print_statistics(summarize(round_ms));
    std::cout << ",\"plans_per_second\":"
              << 1000.0 * static_cast<double>(requests) / summarize(round_ms).mean
              << ",\"stage_latency\":{";
    constexpr std::array<const char *, 4> stage_names{
        "root", "pose", "vq_decoder", "representation"};
    for (std::size_t stage = 0; stage < stage_samples.size(); ++stage) {
        if (stage != 0U) std::cout << ',';
        std::cout << '\"' << stage_names[stage] << "\":";
        print_statistics(summarize(stage_samples[stage]));
    }
    std::cout << "},\"frame_counts\":[";
    for (std::size_t i = 0; i < frame_counts.size(); ++i) {
        if (i != 0U) std::cout << ',';
        std::cout << frame_counts[i];
    }
    std::cout << "],\"selected_tokens\":[";
    for (std::size_t i = 0; i < selected_tokens.size(); ++i) {
        if (i != 0U) std::cout << ',';
        std::cout << selected_tokens[i];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & exception) {
    std::cerr << "motionbricks-cpu-profile: " << exception.what() << '\n';
    return 1;
}
