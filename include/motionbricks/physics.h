#ifndef MOTIONBRICKS_PHYSICS_H
#define MOTIONBRICKS_PHYSICS_H
#include <motionbricks/sonic.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mb_physics mb_physics;
/* Simulation only. SONIC must outlive the session. Each session owns its
   MuJoCo data/history. Caller serializes session calls and destruction.
   No struct layouts cross this ABI. Disabled builds return UNAVAILABLE. */
MB_API mb_status mb_physics_create(mb_sonic * sonic, const char * scene_xml,
    const char * config_file, mb_physics ** output, char * error, uint64_t capacity);
MB_API void mb_physics_free(mb_physics * session);
/* Explicitly clear state/history while retaining the compiled robot model.
   The caller must start again; no reset occurs during ordinary stepping. */
MB_API mb_status mb_physics_reset(mb_physics * session, char * error, uint64_t capacity);
/* Initialize once from a native G1 pose; subsequent changes of reference do
   not reset physical state. Call mb_physics_reset before starting again. */
MB_API mb_status mb_physics_start(mb_physics * session, const float * root, uint64_t root_count,
    const float * local_xyzw, uint64_t rotation_count, char * error, uint64_t capacity);
/* Step one 50 Hz controller tick + four 200 Hz actuator physics steps.
   Source is native G1Skeleton34 at 30 FPS. Time is seconds into this source.
   Beyond its end the final pose is held; future velocities become zero.
   Physical output is 30 XYZ joints, Y-up, in IsaacLab body order.
   Reference output uses the same physical FK, without recentering. */
MB_API mb_status mb_physics_step(mb_physics * session, const float * roots, uint64_t root_count,
    const float * rotations, uint64_t rotation_count, uint32_t frames, double source_time,
    float * actual_xyz, uint64_t actual_count, float * reference_xyz, uint64_t reference_count,
    char * error, uint64_t capacity);
MB_API mb_status mb_physics_skeleton(mb_physics * session, int32_t * parents, uint64_t count,
    char * error, uint64_t capacity);
/* fallen records a detected fall since reset; it does not stop stepping. */
MB_API mb_status mb_physics_status(mb_physics * session, double * time, uint32_t * fallen,
    uint32_t * contacts, char * error, uint64_t capacity);
/* Robot collision geoms only (world/floor and visual-only geoms excluded).
   Stable indices for the session lifetime. Type codes: sphere=2, capsule=3,
   ellipsoid=4, cylinder=5, box=6, convex mesh=7. Sizes are MuJoCo's local
   half-extents/radii; capsule/cylinder axis is local Z. Names identify bodies.
   No borrowed memory or public struct layouts. */
MB_API mb_status mb_physics_collision_count(mb_physics * session, uint32_t * count,
    char * error, uint64_t error_capacity);
MB_API mb_status mb_physics_collision_shape(mb_physics * session, uint32_t index,
    uint32_t * type, float * size, uint64_t size_count, char * name, uint64_t name_capacity,
    char * error, uint64_t error_capacity);
/* Mesh hull triangle vertices in compiled geom-local XYZ, without scaling or
   basis conversion. NULL/zero queries float count; primitives return count 0. */
MB_API mb_status mb_physics_collision_triangles(mb_physics * session, uint32_t index,
    float * xyz, uint64_t capacity, uint64_t * count, char * error, uint64_t error_capacity);
/* One world XYZ + XYZW per shape (7 floats). World is viewer Y-up, but local
   geometry remains in MuJoCo's geom frame: rotation includes that basis change.
   These are actual geom_xpos/geom_xmat, not skeleton-derived transforms. */
MB_API mb_status mb_physics_collision_transforms(mb_physics * session, float * transforms,
    uint64_t count, char * error, uint64_t error_capacity);
/* Last completed tick diagnostics, copied as doubles. Field 0: encoder 1762;
   1: decoder 994; 2: actions 29; 3: hardware-order motor targets 29;
   4: pre-controller hardware q29,dq29,base WXYZ4,body angular velocity3;
   5: four substeps, each hardware q29,dq29,clipped torque29 (up to 348).
   NULL data / zero capacity queries the required size. No trace before a tick. */
MB_API mb_status mb_physics_trace(mb_physics * session, uint32_t field,
    double * data, uint64_t capacity, uint64_t * count, char * error, uint64_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
