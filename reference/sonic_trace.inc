// Observation-only instrumentation inserted AFTER the official controller
// has computed and published its motor-command buffer. No buffer mutations.
if (const char* trace_path = std::getenv("SONIC_TRACE_FILE")) {
  static std::ofstream trace(trace_path);
  auto history = state_logger_->GetLatest(10, control_dt_, false);
  const auto motor = motor_command_buffer_.GetDataWithTime().data;
  nlohmann::json row;
  row["index"] = history.back().index;
  row["state_time"] = std::chrono::duration<double>(history.back().timestamp_monotonic.time_since_epoch()).count();
  row["command_time"] = std::chrono::duration<double>(motor_command_end_time.time_since_epoch()).count();
  row["frame"] = current_frame_copy;
  row["playing"] = current_play_copy;
  row["mode"] = current_encoder_mode_copy;
  row["encoder_input"] = encoder_engine_->GetInputBuffer();
  row["decoder_input"] = policy_engine_->GetInputBuffer();
  row["tokens"] = encoder_engine_->GetTokenBuffer();
  row["actions"] = policy_engine_->GetActionBuffer();
  row["target_q"] = motor->q_target;
  row["action_scale"] = g1_action_scale;
  row["default_angles"] = default_angles;
  row["apply_heading"] = ComputeApplyDeltaHeading();
  for (const auto& obs : active_encoder_obs_functions_)
    row["encoder_layout"].push_back({{"name",obs.name},{"offset",obs.offset},{"dimension",obs.dimension}});
  for (const auto& h : history)
    row["history"].push_back({{"index",h.index},{"base_quat",h.base_quat},
      {"base_ang_vel",h.base_ang_vel},{"body_q",h.body_q},{"body_dq",h.body_dq},{"last_action",h.last_action}});
  trace << row.dump() << '\n';
  if (!trace) throw std::runtime_error("SONIC diagnostic trace write failed");
}
