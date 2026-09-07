// A critically damped focus filter, independent of rendering and skeleton
// mutation. Position and look-at share this anchor so they cannot fight.
export class CameraFollow {
  constructor() { this.position = null; this.velocity = [0, 0, 0]; }
  reset(target) { this.position = [...target]; this.velocity = [0, 0, 0]; }
  update(target, dt) {
    if (!target.every(Number.isFinite)) return this.position;
    if (!this.position) this.reset(target);
    dt = Number.isFinite(dt) ? Math.max(0, Math.min(dt, 0.1)) : 0;
    if (dt === 0) return this.position;
    for (let axis = 0; axis < 3; axis++) {
      const omega = axis === 1 ? 2.2 : 4.0;
      const maxSpeed = axis === 1 ? 0.8 : 6.0;
      const decay = Math.exp(-omega * dt);
      const displacement = this.position[axis] - target[axis];
      const impulse = this.velocity[axis] + omega * displacement;
      const next = target[axis] + (displacement + impulse * dt) * decay;
      const step = Math.max(-maxSpeed * dt, Math.min(next - this.position[axis], maxSpeed * dt));
      this.position[axis] += step;
      this.velocity[axis] = Math.max(-maxSpeed, Math.min((this.velocity[axis] - omega * impulse * dt) * decay, maxSpeed));
    }
    return this.position;
  }
}
