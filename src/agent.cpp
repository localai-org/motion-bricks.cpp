#include "agent.hpp"

#include "handles.hpp"
#include "motion_rep.hpp"
#include "planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace motionbricks::detail {
namespace {

using mat3 = std::array<float, 9>;

mat3 multiply(const mat3 & a, const mat3 & b) {
    mat3 result{};
    for (unsigned row = 0; row < 3U; ++row)
        for (unsigned column = 0; column < 3U; ++column)
            for (unsigned k = 0; k < 3U; ++k)
                result[row * 3U + column] += a[row * 3U + k] * b[k * 3U + column];
    return result;
}

mat3 transpose(const mat3 & value) {
    return {value[0],value[3],value[6],value[1],value[4],value[7],value[2],value[5],value[8]};
}

mat3 y_rotation(float angle) {
    const float cosine = std::cos(angle), sine = std::sin(angle);
    return {cosine,0.0F,sine, 0.0F,1.0F,0.0F, -sine,0.0F,cosine};
}

std::array<float, 3> transform(const mat3 & matrix, const float * value) {
    return {matrix[0]*value[0] + matrix[1]*value[1] + matrix[2]*value[2],
            matrix[3]*value[0] + matrix[4]*value[1] + matrix[5]*value[2],
            matrix[6]*value[0] + matrix[7]*value[1] + matrix[8]*value[2]};
}

mat3 quaternion_matrix(const float * xyzw) {
    const float x=xyzw[0], y=xyzw[1], z=xyzw[2], w=xyzw[3];
    const float scale=2.0F/(x*x+y*y+z*z+w*w);
    return {1.0F-scale*(y*y+z*z),scale*(x*y-z*w),scale*(x*z+y*w),
            scale*(x*y+z*w),1.0F-scale*(x*x+z*z),scale*(y*z-x*w),
            scale*(x*z-y*w),scale*(y*z+x*w),1.0F-scale*(x*x+y*y)};
}

mat3 cont6d_matrix(const float * value) {
    std::array<float,3> x{value[0],value[1],value[2]}, y_raw{value[3],value[4],value[5]};
    const auto normalize=[](std::array<float,3> & v) {
        const float length=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        for(float & item:v)item/=length;
    };
    normalize(x);
    std::array<float,3> z{x[1]*y_raw[2]-x[2]*y_raw[1],x[2]*y_raw[0]-x[0]*y_raw[2],
                          x[0]*y_raw[1]-x[1]*y_raw[0]};
    normalize(z);
    const std::array<float,3> y{z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]};
    return {x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]};
}

void matrix_cont6d(const mat3 & matrix, float * value) {
    value[0]=matrix[0];value[1]=matrix[3];value[2]=matrix[6];
    value[3]=matrix[1];value[4]=matrix[4];value[5]=matrix[7];
}

std::array<float, 4> matrix_xyzw(const mat3 & matrix) {
    std::array<float, 4> wxyz{};
    const float trace = matrix[0] + matrix[4] + matrix[8];
    if (trace > 0.0F) {
        const float s = 2.0F * std::sqrt(std::max(0.0F, trace + 1.0F));
        wxyz = {0.25F*s, (matrix[7]-matrix[5])/s, (matrix[2]-matrix[6])/s,
                (matrix[3]-matrix[1])/s};
    } else if (matrix[0] > matrix[4] && matrix[0] > matrix[8]) {
        const float s = 2.0F * std::sqrt(std::max(0.0F, 1.0F+matrix[0]-matrix[4]-matrix[8]));
        wxyz = {(matrix[7]-matrix[5])/s,0.25F*s,(matrix[1]+matrix[3])/s,(matrix[2]+matrix[6])/s};
    } else if (matrix[4] > matrix[8]) {
        const float s = 2.0F * std::sqrt(std::max(0.0F, 1.0F+matrix[4]-matrix[0]-matrix[8]));
        wxyz = {(matrix[2]-matrix[6])/s,(matrix[1]+matrix[3])/s,0.25F*s,(matrix[5]+matrix[7])/s};
    } else {
        const float s = 2.0F * std::sqrt(std::max(0.0F, 1.0F+matrix[8]-matrix[0]-matrix[4]));
        wxyz = {(matrix[3]-matrix[1])/s,(matrix[2]+matrix[6])/s,(matrix[5]+matrix[7])/s,0.25F*s};
    }
    const float norm = std::sqrt(wxyz[0]*wxyz[0]+wxyz[1]*wxyz[1]+wxyz[2]*wxyz[2]+wxyz[3]*wxyz[3]);
    for (float & value : wxyz) value /= norm;
    return {wxyz[1],wxyz[2],wxyz[3],wxyz[0]};
}

float wrap(float value) {
    constexpr float pi = 3.14159265358979323846F;
    return std::remainder(value, 2.0F*pi);
}

float spring(float current, float velocity, float target, float half_life, float time) {
    constexpr float ln2 = 0.69314718056F;
    const float y = (4.0F * ln2) / (half_life + 1.0e-5F) / 2.0F;
    const float x = y * time;
    const float decay = 1.0F / (1.0F + x + 0.48F*x*x + 0.235F*x*x*x);
    const float j0 = current - target;
    const float j1 = velocity + j0*y;
    return (j0 + j1*time)*decay + target;
}

struct world_frame {
    float x{};
    float z{};
    float heading{};
};

world_frame canonicalize(const mb_agent & source, const mb_command & source_command,
                         mb_agent & agent, mb_command & command) {
    const auto first=source.context_frames-boundary_frame_count;
    const float * first_root=source.context_root_xyz.data()+first*3U;
    const float * first_rotation=source.context_local_rotations_xyzw.data()+
        first*g1_joint_count*4U;
    const auto root_matrix=quaternion_matrix(first_rotation);
    world_frame frame{first_root[0],first_root[2],std::atan2(root_matrix[2],root_matrix[8])};
    const auto inverse_heading=y_rotation(-frame.heading);
    agent.model=source.model;
    agent.context_frames=source.context_frames;
    agent.context_root_xyz=source.context_root_xyz;
    agent.context_local_rotations_xyzw=source.context_local_rotations_xyzw;
    for(std::uint64_t index=0;index<agent.context_frames;++index) {
        float * root=agent.context_root_xyz.data()+index*3U;
        const float value[3]{root[0]-frame.x,root[1],root[2]-frame.z};
        const auto rotated=transform(inverse_heading,value);
        std::copy(rotated.begin(),rotated.end(),root);
        float * rotation=agent.context_local_rotations_xyzw.data()+index*g1_joint_count*4U;
        const auto canonical=multiply(inverse_heading,quaternion_matrix(rotation));
        const auto xyzw=matrix_xyzw(canonical);
        std::copy(xyzw.begin(),xyzw.end(),rotation);
    }
    command=source_command;
    const auto movement=transform(inverse_heading,source_command.movement_direction.data());
    const auto facing=transform(inverse_heading,source_command.facing_direction.data());
    command.movement_direction=movement;command.facing_direction=facing;
    if(command.has_world_target!=0U) {
        const float target[3]{command.world_target[0]-frame.x,command.world_target[1],
                              command.world_target[2]-frame.z};
        command.world_target=transform(inverse_heading,target);
        command.world_target_heading-=frame.heading;
    }
    return frame;
}

void restore_world(mb_motion & motion, const world_frame & frame) {
    if(motion.frames==0U)return;
    const auto first_matrix=quaternion_matrix(motion.local_rotations_xyzw.data());
    const float first_heading=std::atan2(first_matrix[2],first_matrix[8]);
    const auto output_correction=y_rotation(frame.heading-first_heading);
    const float first_x=motion.root_translations[0],first_z=motion.root_translations[2];
    for(std::uint64_t index=0;index<motion.frames;++index) {
        float * root=motion.root_translations.data()+index*3U;
        const float value[3]{root[0]-first_x,root[1],root[2]-first_z};
        const auto rotated=transform(output_correction,value);
        root[0]=rotated[0]+frame.x;root[1]=rotated[1];root[2]=rotated[2]+frame.z;
        float * rotation=motion.local_rotations_xyzw.data()+index*g1_joint_count*4U;
        const auto world=multiply(output_correction,quaternion_matrix(rotation));
        const auto xyzw=matrix_xyzw(world);std::copy(xyzw.begin(),xyzw.end(),rotation);
    }
    const auto target_correction=y_rotation(frame.heading);
    for(std::uint64_t index=0;index<motion.target_frames;++index) {
        float * root=motion.target_root_translations.data()+index*3U;
        const auto rotated=transform(target_correction,root);
        root[0]=rotated[0]+frame.x;root[1]=rotated[1];root[2]=rotated[2]+frame.z;
        float * rotation=motion.target_local_rotations_xyzw.data()+index*g1_joint_count*4U;
        const auto world=multiply(target_correction,quaternion_matrix(rotation));
        const auto xyzw=matrix_xyzw(world);std::copy(xyzw.begin(),xyzw.end(),rotation);
    }
}

mb_status build_constraints(mb_agent & agent, const mb_command & command,
                            const mb_style & style, transition_constraints & result,
                            std::string & reason) {
    encoded_frames context;
    auto status = encode_context(*agent.model, agent.context_root_xyz,
        agent.context_local_rotations_xyzw, agent.context_frames, context, reason);
    if (status != MB_OK) return status;
    result.allowed_tokens = style.allowed_tokens;

    const float current_x = context.global_root[0], current_z = context.global_root[2];
    const float velocity_x = (context.global_root[5] - current_x) * 30.0F;
    const float velocity_z = (context.global_root[7] - current_z) * 30.0F;
    float direction_x = command.movement_direction[0], direction_z = command.movement_direction[2];
    const float direction_norm = std::hypot(direction_x, direction_z);
    if (direction_norm <= 1.0e-5F) {
        direction_x=command.facing_direction[0]*0.1F;
        direction_z=command.facing_direction[2]*0.1F;
    }
    float travel = command.target_speed >= 0.0F ? command.target_speed*2.0F : style.speed*2.0F;
    if (command.target_speed < 0.0F && style.speed == 0.0F)
        travel=std::hypot(velocity_x,velocity_z)/2.0F;
    if (travel <= 0.1F) travel=0.0F;
    const float target_x = command.has_world_target != 0U ? command.world_target[0]
        : current_x + travel * direction_x;
    const float target_z = command.has_world_target != 0U ? command.world_target[2]
        : current_z + travel * direction_z;

    const float current_heading = std::atan2(context.global_root[4], context.global_root[3]);
    const float next_heading = std::atan2(context.global_root[9], context.global_root[8]);
    const float heading_velocity = wrap(next_heading - current_heading) * 30.0F;
    float desired_heading = command.has_world_target != 0U ? command.world_target_heading
        : std::atan2(command.facing_direction[0], command.facing_direction[2]);
    desired_heading = current_heading + wrap(desired_heading - current_heading);

    std::array<float,4> start_root_x{},start_root_z{},start_heading{};
    std::array<float, 4> target_root_x{}, target_root_z{}, target_heading{};
    for (unsigned frame = 0; frame < 4U; ++frame) {
        const float start_time=static_cast<float>(frame)/30.0F;
        const float time = 1.0F + static_cast<float>(frame) / 30.0F;
        start_root_x[frame]=spring(current_x,velocity_x,target_x,0.8F,start_time);
        start_root_z[frame]=spring(current_z,velocity_z,target_z,0.8F,start_time);
        start_heading[frame]=spring(current_heading,heading_velocity,desired_heading,0.17F,start_time);
        target_root_x[frame] = spring(current_x, velocity_x, target_x, 0.8F, time);
        target_root_z[frame] = spring(current_z, velocity_z, target_z, 0.8F, time);
        target_heading[frame] = spring(current_heading, heading_velocity, desired_heading, 0.17F, time);
    }
    if (style.speed==0.0F && command.target_speed<0.0F && command.has_world_target==0U) {
        std::fill(target_root_x.begin(),target_root_x.end(),target_root_x.front());
        std::fill(target_root_z.begin(),target_root_z.end(),target_root_z.front());
    }

    // Upstream applies the spring's t=[0, 3/fps] path to the source context
    // before encoding sparse constraints. This keeps incoming momentum while
    // allowing the one-second target to change immediately.
    constexpr std::size_t rotation_offset=(g1_joint_count-1U)*3U;
    for (std::uint32_t frame=0;frame<4U;++frame) {
        const float old_heading=std::atan2(context.global_root[frame*global_root_width+4U],
                                           context.global_root[frame*global_root_width+3U]);
        const auto correction=y_rotation(start_heading[frame]-old_heading);
        auto * pose=context.poses.data()+static_cast<std::size_t>(frame)*external_pose_width;
        for(std::uint32_t joint=1;joint<g1_joint_count;++joint) {
            const auto position=transform(correction,pose+(joint-1U)*3U);
            std::copy(position.begin(),position.end(),pose+(joint-1U)*3U);
        }
        for(std::uint32_t joint=0;joint<g1_joint_count;++joint) {
            float * six=pose+rotation_offset+joint*6U;
            const auto rotated=multiply(correction,cont6d_matrix(six));
            matrix_cont6d(rotated,six);
        }
        auto * global=context.global_root.data()+frame*global_root_width;
        global[0]=start_root_x[frame];global[2]=start_root_z[frame];
        global[3]=std::cos(start_heading[frame]);global[4]=std::sin(start_heading[frame]);
    }
    for(std::uint32_t frame=0;frame<3U;++frame) {
        const float * current=context.global_root.data()+frame*global_root_width;
        const float * next=current+global_root_width;
        float * local=context.local_root.data()+frame*local_root_width;
        local[0]=wrap(start_heading[frame+1U]-start_heading[frame])*30.0F;
        local[1]=(next[0]-current[0])*30.0F;local[2]=(next[2]-current[2])*30.0F;local[3]=current[1];
    }
    std::copy(context.global_root.begin(), context.global_root.end(), result.global_root.begin());
    std::copy(context.local_root.begin(), context.local_root.end(), result.local_root.begin());
    std::copy(context.poses.begin(), context.poses.end(), result.poses.begin());
    result.has_local_root[3] = 0U;
    const std::uint32_t first = style.frames == 4U ? 0U
        : static_cast<std::uint32_t>(command.seed % (style.frames - 4U));
    for (std::uint32_t frame = 0; frame < 4U; ++frame) {
        const auto source_frame = first + frame;
        const float correction = wrap(target_heading[frame] - style.global_headings[source_frame]);
        const mat3 correction_matrix = y_rotation(correction);
        auto * global = result.global_root.data() + (frame + 4U) * global_root_width;
        const float * style_root_joint = style.global_joint_positions.data() +
            (static_cast<std::size_t>(source_frame) * g1_joint_count) * 3U;
        global[0] = target_root_x[frame];
        global[1] = style.global_root_positions[source_frame * 3U + 1U] + style_root_joint[1];
        global[2] = target_root_z[frame];
        global[3] = std::cos(target_heading[frame]); global[4] = std::sin(target_heading[frame]);
        std::copy_n(global, 3U, result.target_root_translations.data() + frame * 3U);
        float * pose = result.poses.data() + (frame + 4U) * external_pose_width;
        for (std::uint32_t joint = 1; joint < g1_joint_count; ++joint) {
            const float * source = style.global_joint_positions.data() +
                (static_cast<std::size_t>(source_frame) * g1_joint_count + joint) * 3U;
            const auto position = transform(correction_matrix, source);
            std::copy(position.begin(), position.end(), pose + (joint - 1U) * 3U);
        }
        std::array<mat3, g1_joint_count> target_global_rotations{};
        for (std::uint32_t joint = 0; joint < g1_joint_count; ++joint) {
            const float * source = style.global_joint_rotations.data() +
                (static_cast<std::size_t>(source_frame) * g1_joint_count + joint) * 9U;
            mat3 source_matrix{};
            std::copy_n(source, 9, source_matrix.begin());
            const auto matrix = multiply(correction_matrix, source_matrix);
            target_global_rotations[joint] = matrix;
            float * six = pose + rotation_offset + joint * 6U;
            six[0]=matrix[0]; six[1]=matrix[3]; six[2]=matrix[6];
            six[3]=matrix[1]; six[4]=matrix[4]; six[5]=matrix[7];
        }
        for (std::uint32_t joint = 0; joint < g1_joint_count; ++joint) {
            mat3 local = target_global_rotations[joint];
            const auto parent = agent.model->joint_parents[joint];
            if (parent >= 0)
                local = multiply(transpose(target_global_rotations[static_cast<std::size_t>(parent)]),
                                 target_global_rotations[joint]);
            const auto xyzw = matrix_xyzw(local);
            std::copy(xyzw.begin(), xyzw.end(), result.target_local_rotations_xyzw.begin() +
                static_cast<std::ptrdiff_t>((static_cast<std::size_t>(frame) * g1_joint_count + joint) * 4U));
        }
    }
    for (std::uint32_t frame = 0; frame < 3U; ++frame) {
        const float * current = result.global_root.data() + (frame + 4U) * global_root_width;
        const float * next = current + global_root_width;
        float * local = result.local_root.data() + (frame + 4U) * local_root_width;
        local[0] = wrap(target_heading[frame + 1U] - target_heading[frame]) * 30.0F;
        local[1] = (next[0] - current[0]) * 30.0F;
        local[2] = (next[2] - current[2]) * 30.0F;
        local[3] = current[1];
    }
    std::copy_n(result.local_root.data() + 6U * local_root_width, local_root_width,
                result.local_root.data() + 7U * local_root_width);
    result.local_root[7U * local_root_width + 3U] = result.global_root[7U * global_root_width + 1U];
    return MB_OK;
}

} // namespace

mb_status seed_agent_from_style(mb_agent & agent, const mb_style & style,
                                std::string & reason) {
    if (agent.model == nullptr || style.frames < 4U) {
        reason = "agent or initial style is invalid";
        return MB_INVALID_ARGUMENT;
    }
    agent.context_frames = 4U;
    agent.context_root_xyz.resize(12U);
    agent.context_local_rotations_xyzw.resize(4U * g1_joint_count * 4U);
    for (std::uint32_t frame = 0; frame < 4U; ++frame) {
        const float * root_joint = style.global_joint_positions.data() + frame * g1_joint_count * 3U;
        for (unsigned axis = 0; axis < 3U; ++axis)
            agent.context_root_xyz[frame * 3U + axis] =
                style.global_root_positions[frame * 3U + axis] + root_joint[axis];
        for (std::uint32_t joint = 0; joint < g1_joint_count; ++joint) {
            const float * source = style.global_joint_rotations.data() +
                (static_cast<std::size_t>(frame) * g1_joint_count + joint) * 9U;
            mat3 global{}; std::copy_n(source, 9, global.begin());
            const auto parent = agent.model->joint_parents[joint];
            mat3 local = global;
            if (parent >= 0) {
                const float * parent_source = style.global_joint_rotations.data() +
                    (static_cast<std::size_t>(frame) * g1_joint_count +
                     static_cast<std::size_t>(parent)) * 9U;
                mat3 parent_global{}; std::copy_n(parent_source, 9, parent_global.begin());
                local = multiply(transpose(parent_global), global);
            }
            const auto xyzw = matrix_xyzw(local);
            std::copy(xyzw.begin(), xyzw.end(), agent.context_local_rotations_xyzw.begin() +
                static_cast<std::ptrdiff_t>((static_cast<std::size_t>(frame) * g1_joint_count + joint) * 4U));
        }
    }
    agent.current_motion.reset();
    agent.current_frame = 0U;
    agent.initial_style = &style;
    return MB_OK;
}

mb_status plan_agent_impl(mb_agent & agent, const mb_command & command,
                          mb_motion & output, transition_trace * trace,
                          std::string & reason) {
    const mb_style * style = command.style != nullptr ? command.style : agent.initial_style;
    if (agent.model == nullptr || style == nullptr) {
        reason = "agent plan requires a style";
        return MB_INVALID_ARGUMENT;
    }
    if (agent.context_frames < 4U) {
        const auto status = seed_agent_from_style(agent, *style, reason);
        if (status != MB_OK) return status;
    }
    mb_agent canonical_agent;
    mb_command canonical_command;
    const auto frame=canonicalize(agent,command,canonical_agent,canonical_command);
    transition_constraints constraints;
    auto status = build_constraints(canonical_agent, canonical_command, *style, constraints, reason);
    if (status != MB_OK) return status;
    status = run_transition(*agent.model, constraints, output, nullptr, trace, reason);
    if (status != MB_OK) return status;
    output.target_frames = 4U;
    output.target_root_translations.assign(constraints.target_root_translations.begin(),
                                           constraints.target_root_translations.end());
    output.target_local_rotations_xyzw.assign(constraints.target_local_rotations_xyzw.begin(),
                                              constraints.target_local_rotations_xyzw.end());
    restore_world(output,frame);
    return MB_OK;
}

mb_status plan_agent(mb_agent & agent, const mb_command & command,
                     mb_motion & output, std::string & reason) {
    return plan_agent_impl(agent, command, output, nullptr, reason);
}

mb_status plan_agent_trace(mb_agent & agent, const mb_command & command,
                           mb_motion & output, transition_trace & trace,
                           std::string & reason) {
    return plan_agent_impl(agent, command, output, &trace, reason);
}

mb_status advance_agent(mb_agent & agent, std::uint32_t frames,
                        std::string & reason) {
    if (!agent.current_motion || agent.current_motion->frames == 0U) {
        reason = "agent has no planned motion";
        return MB_INVALID_ARGUMENT;
    }
    agent.current_frame = std::min<std::uint32_t>(
        agent.current_frame + frames, static_cast<std::uint32_t>(agent.current_motion->frames - 1U));
    agent.context_frames = 4U;
    agent.context_root_xyz.resize(12U);
    agent.context_local_rotations_xyzw.resize(4U * g1_joint_count * 4U);
    for (std::uint32_t frame = 0; frame < 4U; ++frame) {
        const auto source = std::min<std::uint64_t>(agent.current_frame + frame,
                                                   agent.current_motion->frames - 1U);
        std::copy_n(agent.current_motion->root_translations.data() + source * 3U, 3,
                    agent.context_root_xyz.data() + frame * 3U);
        std::copy_n(agent.current_motion->local_rotations_xyzw.data() + source * g1_joint_count * 4U,
                    g1_joint_count * 4U,
                    agent.context_local_rotations_xyzw.data() + frame * g1_joint_count * 4U);
    }
    return MB_OK;
}

} // namespace motionbricks::detail
