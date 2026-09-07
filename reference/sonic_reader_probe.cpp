// Test harness around NVIDIA's actual CSV reader, not a replacement parser.
// Compile with ASan/UBSan and the pinned deployment include directory.
#include <memory>
#include <cmath>
#include <nlohmann/json.hpp>
#include "motion_data_reader.hpp"

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: sonic-reader-probe MOTIONS OUTPUT.json");
        MotionDataReader reader;
        if (!reader.ReadFromCSV(argv[1]) || reader.motions.size() != 1)
            throw std::runtime_error("expected exactly one valid motion");
        auto &m = *reader.motions.at(0);
        if (m.GetNumJoints() != 29 || m.GetNumBodies() != 30 ||
            m.GetNumBodyQuaternions() != 30 || m.timesteps < 3)
            throw std::runtime_error("unexpected native export shape");
        for (int i = 0; i < 30; ++i)
            if (m.BodyPartIndexes().at(i) != i) throw std::runtime_error("wrong body order");
        nlohmann::json result;
        result["frames"] = m.timesteps;
        result["hardware_to_isaac_gather"] = mujoco_to_isaaclab;
        for (int f = 0; f < m.timesteps; ++f) {
            for (int j = 0; j < 29; ++j) {
                if (!std::isfinite(m.JointPositions(f)[j]) || !std::isfinite(m.JointVelocities(f)[j]))
                    throw std::runtime_error("non-finite joint data");
                result["joint_pos"].push_back(m.JointPositions(f)[j]);
                result["joint_vel"].push_back(m.JointVelocities(f)[j]);
            }
            for (int b = 0; b < 30; ++b) {
                double norm = 0;
                for (double q : m.BodyQuaternions(f)[b]) norm += q*q;
                if (!std::isfinite(norm) || std::abs(norm - 1) > 1e-6)
                    throw std::runtime_error("invalid body quaternion");
                result["body_pos"].push_back(m.BodyPositions(f)[b]);
                result["body_quat"].push_back(m.BodyQuaternions(f)[b]);
                result["body_lin_vel"].push_back(m.BodyLinVelocities(f)[b]);
                result["body_ang_vel"].push_back(m.BodyAngVelocities(f)[b]);
            }
        }
        std::ofstream stream(argv[2]);
        stream << result.dump() << '\n';
        if (!stream) throw std::runtime_error("could not write reader result");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
