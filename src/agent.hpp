#pragma once

#include <motionbricks/motionbricks.h>

#include <cstdint>
#include <string>

struct mb_agent;
struct mb_command;
struct mb_motion;
struct mb_style;

namespace motionbricks::detail {
struct transition_trace;
}

namespace motionbricks::detail {

mb_status seed_agent_from_style(mb_agent & agent, const mb_style & style,
                                std::string & reason);
mb_status plan_agent(mb_agent & agent, const mb_command & command,
                     mb_motion & output, std::string & reason);
mb_status plan_agent_trace(mb_agent & agent, const mb_command & command,
                           mb_motion & output, transition_trace & trace,
                           std::string & reason);
mb_status advance_agent(mb_agent & agent, std::uint32_t frames,
                        std::string & reason);

} // namespace motionbricks::detail
