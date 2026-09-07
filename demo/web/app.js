import * as THREE from './vendor/three.module.min.js';
import {CameraFollow} from './camera-follow.js';
import {StreamClient} from './stream-client.js';
import {CollisionGeometry} from './collision-geometry.js';

const viewport = document.querySelector('#viewport');
const styleSelect = document.querySelector('#style-select');
const statusElement = document.querySelector('#status');
const planInfo = document.querySelector('#plan-info');
const targetInfo = document.querySelector('#target-info');
const testResult = document.querySelector('#test-result');
const showAllTargets = document.querySelector('#show-all-targets');
const targetFrame = document.querySelector('#target-frame');
const targetFrameLabel = document.querySelector('#target-frame-label');
const resetCamera = document.querySelector('#reset-camera');
const replayControls = document.querySelector('#replay-controls');
const replayToggle = document.querySelector('#replay-toggle');
const replayFrame = document.querySelector('#replay-frame');
const replayFrameLabel = document.querySelector('#replay-frame-label');
const comparisonPlanRow = document.querySelector('#comparison-plan-row');
const comparisonPlan = document.querySelector('#comparison-plan');
const jumpButton = document.querySelector('#jump');
const kimodoControls = document.querySelector('#kimodo-controls');
const kimodoSelect = document.querySelector('#kimodo-select');
const kimodoPlay = document.querySelector('#kimodo-play');
const kimodoUpload = document.querySelector('#kimodo-upload');
const kimodoFile = document.querySelector('#kimodo-file');
const kimodoUploadStatus = document.querySelector('#kimodo-upload-status');
let uploadingAnimation = false;
const kimodoProgress = document.querySelector('#kimodo-progress');
const kimodoPhase = document.querySelector('#kimodo-phase');
const kimodoProgressBar = document.querySelector('#kimodo-progress-bar');
const kimodoProgressLabel = document.querySelector('#kimodo-progress-label');
const query = new URLSearchParams(location.search);

const state = {
  meta: null, session: '', motion: null, targets: null, playhead: 0,
  lastTime: performance.now(), move: [0, 0], facing: [0, 1], keys: new Set(),
  padKey: '', pending: false, replanQueued: false, seed: 10, style: '', rig: null, targetRigs: [],
  generatedPath: null, targetPath: null,
  replay: null, replayPlaying: true, replayPlan: -2,
  physics: null, physicsTime: 0,
  live: null,
  stream: null,
  comparison: null, comparisonPlan: -1, comparisonShowcase: null,
  comparisonVisiblePlan: -1, nativeRig: null, errorLines: null,
  kimodo: null, jumpActive: false, controlsLocked: false, qaPaused: false,
};

const modeNames = [
  'idle', 'slow_walk', 'walk', 'hand_crawling', 'walk_boxing', 'elbow_crawling',
  'stealth_walk', 'injured_walk', 'walk_stealth', 'walk_happy_dance', 'walk_zombie',
  'walk_gun', 'walk_scared', 'walk_left', 'walk_right',
];
const controllerCadenceFrames = 16;
const jumpTargetSpeed = 5.0;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x090b10, 0.045);
const camera = new THREE.PerspectiveCamera(42, 1, 0.02, 100);
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.append(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xcde4ff, 0x253226, 2.5));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
keyLight.position.set(3, 7, 4);
keyLight.castShadow = true;
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x67e8c3, 1.2);
rimLight.position.set(-4, 3, -3);
scene.add(rimLight);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshStandardMaterial({color: 0x0d121a, roughness: 0.96, metalness: 0.02}),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.012;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(24, 48, 0x3a4b5d, 0x202b38);
grid.position.y = 0;
scene.add(grid);

const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 8, 1, false);
const sphereGeometry = new THREE.SphereGeometry(1, 14, 10);
const diamondGeometry = new THREE.OctahedronGeometry(1, 0);
const up = new THREE.Vector3(0, 1, 0);
const startPoint = new THREE.Vector3();
const endPoint = new THREE.Vector3();
const direction = new THREE.Vector3();
const midpoint = new THREE.Vector3();

function labelSprite(text, color, opacity) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  context.font = '700 30px system-ui';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = color;
  context.fillText(text, 128, 32);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, transparent: true, opacity, depthTest: false}));
  sprite.scale.set(0.5, 0.125, 1);
  sprite.renderOrder = 20;
  return sprite;
}

class SkeletonRig {
  constructor(joints, options) {
    this.bones = [];
    this.jointMeshes = [];
    this.segments = [];
    this.group = new THREE.Group();
    this.radius = options.radius;
    this.jointRadius = options.jointRadius;
    this.rootRadius = options.rootRadius;
    this.labelHeight = options.labelHeight ?? 0.34;
    this.labelOffset = options.labelOffset ?? 0;
    this.boneMaterial = new THREE.MeshStandardMaterial({
      color: options.color, emissive: options.emissive, emissiveIntensity: options.emissiveIntensity,
      roughness: 0.42, transparent: options.opacity < 1, opacity: options.opacity,
      depthWrite: options.opacity >= 0.95, depthTest: options.depthTest ?? true,
    });
    this.jointMaterial = new THREE.MeshStandardMaterial({
      color: options.jointColor, emissive: options.emissive, emissiveIntensity: options.emissiveIntensity,
      roughness: 0.35, transparent: options.opacity < 1, opacity: options.opacity,
      depthWrite: options.opacity >= 0.95, depthTest: options.depthTest ?? true,
    });
    this.rootMaterial = new THREE.MeshStandardMaterial({
      color: options.rootColor, emissive: options.rootEmissive, emissiveIntensity: 1.4,
      transparent: options.opacity < 1, opacity: options.opacity, depthWrite: options.opacity >= 0.95,
      depthTest: options.depthTest ?? true,
    });
    for (let index = 0; index < joints.length; index++) {
      const joint = joints[index];
      const bone = new THREE.Bone();
      bone.name = joint.name;
      if (joint.parent < 0) {
        bone.position.set(0, 0, 0);
      } else {
        const parent = joints[joint.parent].position;
        bone.position.set(joint.position[0] - parent[0], joint.position[1] - parent[1], joint.position[2] - parent[2]);
        this.bones[joint.parent].add(bone);
      }
      this.bones.push(bone);
      const geometry = options.diamonds ? diamondGeometry : sphereGeometry;
      const marker = new THREE.Mesh(geometry, index === 0 ? this.rootMaterial : this.jointMaterial);
      marker.castShadow = !options.diamonds;
      marker.renderOrder = options.renderOrder;
      this.jointMeshes.push(marker);
      this.group.add(marker);
      if (joint.parent >= 0) {
        const segment = new THREE.Mesh(cylinderGeometry, this.boneMaterial);
        segment.castShadow = !options.diamonds;
        segment.renderOrder = options.renderOrder;
        this.segments.push({mesh: segment, child: index, parent: joint.parent});
        this.group.add(segment);
      }
    }
    scene.add(this.bones[0]);
    scene.add(this.group);
    this.label = options.label ? labelSprite(options.label, options.labelColor, options.opacity) : null;
    if (this.label) scene.add(this.label);
  }

  pose(roots, rotations, frame, frameCount) {
    const index = Math.min(frameCount - 1, Math.max(0, frame));
    this.bones[0].position.fromArray(roots, index * 3);
    for (let joint = 0; joint < this.bones.length; joint++) {
      this.bones[joint].quaternion.fromArray(rotations, (index * this.bones.length + joint) * 4);
    }
    this.updateGeometry();
  }

  poseInterpolated(roots, rotations, frame, frameCount) {
    const at=THREE.MathUtils.clamp(frame,0,frameCount-1), first=Math.floor(at),next=Math.min(first+1,frameCount-1),u=at-first;
    this.bones[0].position.fromArray(roots,first*3).lerp(new THREE.Vector3().fromArray(roots,next*3),u);
    const q=new THREE.Quaternion();
    for(let j=0;j<this.bones.length;j++)this.bones[j].quaternion.fromArray(rotations,(first*this.bones.length+j)*4)
      .slerp(q.fromArray(rotations,(next*this.bones.length+j)*4),u);
    this.updateGeometry();
  }

  poseWorld(positions, offset = 0) {
    for (let joint = 0; joint < this.jointMeshes.length; joint++) {
      const marker = this.jointMeshes[joint];
      marker.position.fromArray(positions, offset + joint * 3);
      marker.scale.setScalar(joint === 0 ? this.rootRadius : this.jointRadius);
    }
    for (const segment of this.segments) {
      startPoint.copy(this.jointMeshes[segment.parent].position);
      endPoint.copy(this.jointMeshes[segment.child].position);
      direction.subVectors(endPoint, startPoint);
      const length = direction.length();
      midpoint.addVectors(startPoint, endPoint).multiplyScalar(0.5);
      segment.mesh.position.copy(midpoint);
      segment.mesh.quaternion.setFromUnitVectors(up, direction.normalize());
      segment.mesh.scale.set(this.radius, length, this.radius);
    }
    if (this.label) {
      endPoint.copy(this.jointMeshes[0].position);
      this.label.position.set(endPoint.x + this.labelOffset, endPoint.y + this.labelHeight, endPoint.z);
    }
  }

  updateGeometry() {
    this.bones[0].updateMatrixWorld(true);
    for (let index = 0; index < this.bones.length; index++) {
      this.bones[index].getWorldPosition(endPoint);
      const marker = this.jointMeshes[index];
      marker.position.copy(endPoint);
      const radius = index === 0 ? this.rootRadius : this.jointRadius;
      marker.scale.setScalar(radius);
    }
    for (const segment of this.segments) {
      this.bones[segment.parent].getWorldPosition(startPoint);
      this.bones[segment.child].getWorldPosition(endPoint);
      direction.subVectors(endPoint, startPoint);
      const length = direction.length();
      midpoint.addVectors(startPoint, endPoint).multiplyScalar(0.5);
      segment.mesh.position.copy(midpoint);
      segment.mesh.quaternion.setFromUnitVectors(up, direction.normalize());
      segment.mesh.scale.set(this.radius, length, this.radius);
    }
    if (this.label) {
      this.bones[0].getWorldPosition(endPoint);
      this.label.position.set(endPoint.x + this.labelOffset, endPoint.y + this.labelHeight, endPoint.z);
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
    this.bones[0].visible = visible;
    if (this.label) this.label.visible = visible;
  }

  setOpacity(opacity) {
    for (const material of [this.boneMaterial, this.jointMaterial, this.rootMaterial]) {
      material.opacity = opacity;
      material.transparent = opacity < 1;
      material.depthWrite = opacity >= 0.95;
    }
    if (this.label) this.label.material.opacity = opacity;
  }
}

function makeLine(color, dashed = false) {
  const material = dashed
    ? new THREE.LineDashedMaterial({color, dashSize: 0.09, gapSize: 0.055, transparent: true, opacity: 0.9})
    : new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.72});
  const line = new THREE.Line(new THREE.BufferGeometry(), material);
  line.renderOrder = 1;
  scene.add(line);
  return line;
}

function setGroundPath(line, roots, frames) {
  const points = [];
  for (let frame = 0; frame < frames; frame++) {
    points.push(new THREE.Vector3(roots[frame * 3], 0.018, roots[frame * 3 + 2]));
  }
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(points);
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}

function makeSkeletons(joints) {
  state.rig = new SkeletonRig(joints, {
    color: 0x55efc4, jointColor: 0xd9fff3, rootColor: 0xffd166,
    emissive: 0x0c5b49, rootEmissive: 0x6a3b00, emissiveIntensity: 0.8,
    radius: 0.022, jointRadius: 0.034, rootRadius: 0.062,
    opacity: 1, diamonds: false, renderOrder: 5,
  });
  const targetColors = [
    {bone: 0xffc857, joint: 0xffdc8a, root: 0xffb703, emissive: 0x604000, label: '#ffd166'},
    {bone: 0xff8c42, joint: 0xffb06b, root: 0xff6b1a, emissive: 0x6f2600, label: '#ff9a62'},
    {bone: 0xf04452, joint: 0xff7b84, root: 0xd62839, emissive: 0x65000b, label: '#ff6975'},
    {bone: 0xe11d9a, joint: 0xff71c8, root: 0xb40078, emissive: 0x5d003f, label: '#ff66c4'},
  ];
  for (let frame = 0; frame < 4; frame++) {
    const palette = targetColors[frame];
    state.targetRigs.push(new SkeletonRig(joints, {
      color: palette.bone, jointColor: palette.joint, rootColor: palette.root,
      emissive: palette.emissive, rootEmissive: palette.emissive, emissiveIntensity: 1,
      radius: 0.018, jointRadius: 0.032, rootRadius: 0.056, opacity: 0.9, diamonds: true,
      depthTest: false, renderOrder: 8 + frame, label: `T${frame}`, labelColor: palette.label,
      labelHeight: 0.34 + (3 - frame) * 0.11,
    }));
  }
  state.generatedPath = makeLine(0x43e8bf);
  state.targetPath = makeLine(0xff6b3d, true);
}

function makeComparisonSkeletons(joints, firstLabel = 'UPSTREAM', secondLabel = 'NATIVE') {
  state.rig = new SkeletonRig(joints, {
    color: 0x55efc4, jointColor: 0xd9fff3, rootColor: 0xffd166,
    emissive: 0x0c5b49, rootEmissive: 0x6a3b00, emissiveIntensity: 0.8,
    radius: 0.018, jointRadius: 0.029, rootRadius: 0.055,
    opacity: 0.82, diamonds: false, renderOrder: 5, label: firstLabel, labelColor: '#6fffd0', labelOffset: -0.22,
  });
  state.nativeRig = new SkeletonRig(joints, {
    color: 0x65a9ff, jointColor: 0xd7e8ff, rootColor: 0xff5b8f,
    emissive: 0x173d72, rootEmissive: 0x741536, emissiveIntensity: 0.9,
    radius: 0.014, jointRadius: 0.024, rootRadius: 0.046,
    opacity: 0.72, diamonds: true, renderOrder: 8, label: secondLabel, labelColor: '#7ab6ff', labelOffset: 0.22,
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(joints.length * 2 * 3), 3));
  state.errorLines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({color: 0xff456f, transparent: true, opacity: 0.82}));
  state.errorLines.renderOrder = 12;
  scene.add(state.errorLines);
  state.generatedPath = makeLine(0x55efc4);
  state.targetPath = makeLine(0x65a9ff, true);
}

function updateTargetVisibility() {
  const frames = state.targets?.frames ?? 0;
  const maximum = Math.max(0, frames - 1);
  targetFrame.max = String(maximum);
  const selected = THREE.MathUtils.clamp(Number(targetFrame.value), 0, maximum);
  targetFrame.value = String(selected);
  targetFrameLabel.textContent = `T${selected}`;
  const overlay = showAllTargets.checked;
  const overlayOpacities = [0.16, 0.25, 0.42, 0.9];
  for (let frame = 0; frame < state.targetRigs.length; frame++) {
    const exists = frame < frames;
    state.targetRigs[frame].setOpacity(overlay ? overlayOpacities[frame] : 0.9);
    state.targetRigs[frame].setVisible(exists && (overlay || frame === selected));
  }
  state.targetPath.visible = frames > 0;
  targetInfo.textContent = frames > 0
    ? (overlay ? `${frames} consecutive constraints overlaid` : `T${selected} of ${frames} consecutive constraints`)
    : '—';
}

const cameraView = {yaw: 0.68, pitch: 0.24, distance: 4.8, dragging: false, x: 0, y: 0};
const focus = new THREE.Vector3();
const desiredCamera = new THREE.Vector3();
const cameraSubject = new THREE.Vector3();
const cameraFollow = new CameraFollow();
let cameraFollowTime = performance.now();
function resetCameraView() {
  cameraView.yaw = 0.68;
  cameraView.pitch = 0.24;
  cameraView.distance = 4.8;
  cameraFollow.position = null;
  document.documentElement.dataset.cameraYaw = String(cameraView.yaw);
  if (state.keys.size || state.padKey) { updateControl(); schedulePlan(35); }
}
resetCamera.addEventListener('click', resetCameraView);
renderer.domElement.addEventListener('pointerdown', event => {
  cameraView.dragging = true; cameraView.x = event.clientX; cameraView.y = event.clientY;
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointermove', event => {
  if (!cameraView.dragging) return;
  cameraView.yaw -= (event.clientX - cameraView.x) * 0.006;
  cameraView.pitch = THREE.MathUtils.clamp(cameraView.pitch + (event.clientY - cameraView.y) * 0.004, -0.05, 1.05);
  cameraView.x = event.clientX; cameraView.y = event.clientY;
  document.documentElement.dataset.cameraYaw = String(cameraView.yaw);
  if (state.keys.size || state.padKey) { updateControl(); schedulePlan(35); }
});
renderer.domElement.addEventListener('pointerup', event => {
  cameraView.dragging = false; renderer.domElement.releasePointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointercancel', () => { cameraView.dragging = false; });
renderer.domElement.addEventListener('wheel', event => {
  cameraView.distance = THREE.MathUtils.clamp(cameraView.distance * Math.exp(event.deltaY * 0.001), 2.1, 8);
  event.preventDefault();
}, {passive: false});
renderer.domElement.addEventListener('dblclick', resetCameraView);

function resize() {
  const width = Math.max(1, viewport.clientWidth), height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? 'POST' : 'GET', headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
  return value;
}

async function loadReplay() {
  const response = await fetch('/api/replay', {cache: 'no-store'});
  if (!response.ok) throw new Error(`replay download failed: ${response.status} ${response.statusText}`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < 40) throw new Error('replay is shorter than its header');
  const bytes = new Uint8Array(buffer);
  const expectedMagic = [77, 66, 82, 80, 76, 89, 49, 0];
  if (!expectedMagic.every((value, index) => bytes[index] === value)) throw new Error('unsupported replay magic');
  if (new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] !== 4) throw new Error('big-endian browsers are unsupported');
  const view = new DataView(buffer);
  const header = Array.from({length: 8}, (_, index) => view.getUint32(8 + index * 4, true));
  const [version, fps, frames, joints, qposCount, plans, targetFrames, flags] = header;
  if (version !== 1 || flags !== 0 || targetFrames !== 4 || fps < 1 || fps > 1000 ||
      frames < 1 || frames > 1_000_000 || joints < 1 || joints > 64 || qposCount < 1 || qposCount > 256 ||
      plans < 1 || plans > frames) throw new Error('unsupported replay header');
  let offset = 40;
  const take = (Type, count, name) => {
    const size = Type.BYTES_PER_ELEMENT * count;
    if (!Number.isSafeInteger(count) || count < 0 || offset + size > buffer.byteLength) throw new Error(`replay is truncated in ${name}`);
    const result = new Type(buffer, offset, count);
    offset += size;
    return result;
  };
  const replay = {
    fps, frames, joints, qposCount, plans, targetFrames,
    parents: take(Int32Array, joints, 'parents'),
    modes: take(Int32Array, frames, 'modes'),
    framePlans: take(Int32Array, frames, 'frame plans'),
    qpos: take(Float32Array, frames * qposCount, 'qpos'),
    jointPositions: take(Float32Array, frames * joints * 3, 'joint positions'),
    planFrames: take(Uint32Array, plans, 'plan frames'),
    planModes: take(Int32Array, plans, 'plan modes'),
    planValidLengths: take(Uint32Array, plans, 'plan lengths'),
    targetPositions: take(Float32Array, plans * targetFrames * joints * 3, 'target positions'),
  };
  if (offset !== buffer.byteLength) throw new Error(`replay has ${buffer.byteLength - offset} trailing bytes`);
  if (replay.parents[0] !== -1) throw new Error('replay root parent must be -1');
  for (let joint = 1; joint < joints; joint++) {
    if (replay.parents[joint] < 0 || replay.parents[joint] >= joint) throw new Error('replay joint topology is invalid');
  }
  for (let frame = 0; frame < frames; frame++) {
    if (replay.framePlans[frame] < -1 || replay.framePlans[frame] >= plans) throw new Error('replay frame plan is out of bounds');
  }
  for (let plan = 0; plan < plans; plan++) {
    if (replay.planFrames[plan] >= frames || (plan && replay.planFrames[plan] <= replay.planFrames[plan - 1])) {
      throw new Error('replay plan frames are invalid');
    }
  }
  for (const values of [replay.qpos, replay.jointPositions, replay.targetPositions]) {
    for (const value of values) if (!Number.isFinite(value)) throw new Error('replay contains a non-finite value');
  }
  return replay;
}

function installStyles(styles) {
  for (const style of styles) {
    const option = document.createElement('option');
    option.value = style.name;
    option.textContent = `${style.name.replaceAll('_', ' ')} · ${style.speed.toFixed(1)} m/s`;
    styleSelect.append(option);
  }
  state.style = styles.some(item => item.name === 'walk') ? 'walk' : styles[0].name;
  styleSelect.value = state.style;
  styleSelect.addEventListener('change', () => { state.style = styleSelect.value; void requestPlan(); });
}

function installKimodo(clips) {
  for (const clip of clips || []) {
    if ([...kimodoSelect.options].some(option => option.value === clip.id)) continue;
    const option = document.createElement('option');
    option.value = clip.id;
    option.textContent = `${clip.name} · ${clip.duration.toFixed(1)} s · ${clip.id.split('/').at(-1)}`;
    option.title = clip.prompt;
    kimodoSelect.append(option);
  }
  kimodoControls.hidden = false;
  kimodoPlay.disabled = state.controlsLocked || state.jumpActive || !kimodoSelect.value;
}

kimodoUpload.addEventListener('click', () => kimodoFile.click());
kimodoFile.addEventListener('change', async () => {
  const file = kimodoFile.files[0];
  if (!file || uploadingAnimation || state.controlsLocked) return;
  uploadingAnimation = true;
  kimodoUpload.disabled = true;
  kimodoUploadStatus.dataset.error = 'false';
  kimodoUploadStatus.textContent = `Uploading ${file.name}…`;
  try {
    if (file.size > 16 * 1024 * 1024) throw new Error('Animation upload must be at most 16 MiB.');
    const form = new FormData();
    form.append('animation', file);
    const response = await fetch('/api/kimodo/upload', {method: 'POST', body: form});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Upload failed (${response.status}).`);
    installKimodo([result]);
    kimodoSelect.value = result.id;
    kimodoPlay.disabled = state.controlsLocked || state.jumpActive;
    kimodoUploadStatus.textContent = `Imported ${result.name} (${result.duration.toFixed(1)} s). Press Play to start.`;
  } catch (error) {
    kimodoUploadStatus.dataset.error = 'true';
    kimodoUploadStatus.textContent = error.message;
  } finally {
    uploadingAnimation = false;
    kimodoUpload.disabled = state.controlsLocked;
    kimodoFile.value = '';
  }
});

function lockMotionControls(locked) {
  state.controlsLocked = locked;
  styleSelect.disabled = locked;
  jumpButton.disabled = locked || state.jumpActive;
  kimodoSelect.disabled = locked;
  kimodoPlay.disabled = locked || state.jumpActive || !kimodoSelect.value;
  kimodoUpload.disabled = locked || uploadingAnimation;
  document.querySelectorAll('.pad button').forEach(button => { button.disabled = locked; });
  document.documentElement.dataset.controlsLocked = String(locked);
}

function useMotion(response, plannedMove = state.move, playhead = 0, plannedAdvance = 0,
                   plannedJump = false, plannedSpeed = null) {
  state.session = response.session;
  state.style = response.style;
  styleSelect.value = state.style;
  state.motion = response.motion;
  state.jumpActive = plannedJump;
  jumpButton.disabled = plannedJump || state.controlsLocked;
  kimodoPlay.disabled = plannedJump || state.controlsLocked || !kimodoSelect.value;
  state.targets = response.targets;
  state.playhead = Math.min(playhead, response.motion.frames - 1);
  state.lastTime = performance.now();
  const target = (response.targets.frames - 1) * 3;
  const previousTarget = target - 3;
  const jumpTargetDistance = Math.hypot(response.targets.roots[target] - response.motion.roots[0],
    response.targets.roots[target + 2] - response.motion.roots[2]);
  const jumpTargetVelocity = state.meta.fps * Math.hypot(
    response.targets.roots[target] - response.targets.roots[previousTarget],
    response.targets.roots[target + 2] - response.targets.roots[previousTarget + 2]);
  planInfo.textContent = `${response.motion.frames} frames · ${response.style.replaceAll('_', ' ')}` +
    (plannedJump ? ` · jump ${jumpTargetDistance.toFixed(2)} m target · ${jumpTargetVelocity.toFixed(2)} m/s` : '');
  statusElement.textContent = 'Playing';
  document.documentElement.dataset.planSequence = String(Number(document.documentElement.dataset.planSequence || 0) + 1);
  document.documentElement.dataset.plannedMoveX = String(plannedMove[0]);
  document.documentElement.dataset.plannedMoveZ = String(plannedMove[1]);
  document.documentElement.dataset.plannedAdvance = String(plannedAdvance);
  document.documentElement.dataset.controllerCadenceFrames = String(controllerCadenceFrames);
  document.documentElement.dataset.plannedJump = String(plannedJump);
  document.documentElement.dataset.plannedSpeed = plannedSpeed === null ? '' : String(plannedSpeed);
  document.documentElement.dataset.jumpTargetDistance = plannedJump ? String(jumpTargetDistance) : '0';
  document.documentElement.dataset.jumpTargetVelocity = plannedJump ? String(jumpTargetVelocity) : '0';
  setGroundPath(state.generatedPath, response.motion.roots, response.motion.frames);
  setGroundPath(state.targetPath, response.targets.roots, response.targets.frames);
  for (let frame = 0; frame < response.targets.frames; frame++) {
    state.targetRigs[frame].pose(response.targets.roots, response.targets.rotations, frame, response.targets.frames);
  }
  updateTargetVisibility();
}

async function requestPlan(advance = Math.floor(state.playhead), override = null) {
  if(state.stream){state.stream.client.send('control',{style:state.style,move:[...state.move],facing:[...state.facing]});return null;}
  if (!state.session || state.controlsLocked || state.kimodo) return null;
  if (state.live?.enabled && (state.live.busy || state.live.queue.length)) {
    state.live.deferredPlan = {advance: null, override};
    return null;
  }
  // Do not discard the airborne part at the normal 16-frame walk cadence.
  if (state.jumpActive && state.playhead < state.motion.frames - 1) return null;
  if (state.pending) {
    state.replanQueued = true;
    return null;
  }
  state.pending = true;
  statusElement.textContent = 'Planning…';
  const plannedMove = override?.move ?? [...state.move];
  const plannedSpeed = override?.speed ?? null;
  try {
    const response = await api('/api/plan', {
      session: state.session, style: state.style, move: plannedMove, facing: state.facing,
      speed: plannedSpeed ?? undefined, seed: state.seed++, advance,
    });
    // Frame zero of the replacement is the context at `advance`. Keep any
    // playback time that elapsed while inference was running instead of
    // visibly jumping back to that boundary when the response arrives.
    const replacementPlayhead = Math.max(0, state.playhead - advance);
    useMotion(response, plannedMove, replacementPlayhead, advance, override?.jump === true, plannedSpeed);
    return response;
  } finally {
    state.pending = false;
    if (state.replanQueued) {
      state.replanQueued = false;
      queueMicrotask(() => void requestPlan(Math.floor(state.playhead)));
    }
  }
}

async function beginJump() {
  if(state.stream){state.stream.client.send('jump');return;}
  if (!state.motion || state.pending || state.controlsLocked || state.jumpActive) return;
  const source = Math.hypot(state.move[0], state.move[1]) > 1e-6 ? state.move : state.facing;
  const length = Math.hypot(source[0], source[1]);
  const jumpMove = length > 1e-6 ? [source[0] / length, source[1] / length] : [0, 1];
  jumpButton.disabled = true;
  jumpButton.setAttribute('aria-pressed', 'true');
  try {
    await requestPlan(Math.floor(state.playhead), {
      jump: true, move: jumpMove, speed: jumpTargetSpeed,
    });
  } finally {
    jumpButton.disabled = state.jumpActive || state.controlsLocked;
    jumpButton.setAttribute('aria-pressed', 'false');
  }
}

function cameraRelativeMovement(right, forward) {
  const sine = Math.sin(cameraView.yaw), cosine = Math.cos(cameraView.yaw);
  return [right * cosine - forward * sine, -right * sine - forward * cosine];
}

function updateControl() {
  let right = 0, forward = 0;
  const active = key => state.keys.has(key) || state.padKey === key;
  if (active('w')) forward += 1;
  if (active('s')) forward -= 1;
  if (active('a')) right -= 1;
  if (active('d')) right += 1;
  const length = Math.hypot(right, forward);
  if (length > 0) { right /= length; forward /= length; }
  const [x, z] = cameraRelativeMovement(right, forward);
  if (length > 0) state.facing = [x, z];
  state.move = [x, z];
  document.documentElement.dataset.controlRight = String(right);
  document.documentElement.dataset.controlForward = String(forward);
  document.documentElement.dataset.cameraYaw = String(cameraView.yaw);
  document.querySelectorAll('.pad button').forEach(button => {
    const pressed = active(button.dataset.key);
    button.classList.toggle('active', pressed);
    button.setAttribute('aria-pressed', String(pressed));
  });
}

function currentPose() {
  const frame = THREE.MathUtils.clamp(Math.floor(state.playhead), 0, state.motion.frames - 1);
  return {
    frame,
    root: state.motion.roots.slice(frame * 3, frame * 3 + 3),
    rotations: state.motion.rotations.slice(frame * state.motion.joints * 4, (frame + 1) * state.motion.joints * 4),
  };
}

async function beginKimodo() {
  if(state.stream){if(kimodoSelect.value)state.stream.client.send('play_clip',{clip:kimodoSelect.value});return;}
  if (!state.motion || state.pending || state.controlsLocked || state.jumpActive || !kimodoSelect.value) return;
  if (state.live?.enabled && (state.live.busy || state.live.queue.length)) {
    state.live.deferredKimodo = true;
    return;
  }
  const resume = {move: [...state.move], facing: [...state.facing], style: state.style, padKey: state.padKey};
  clearTimeout(controlTimer);
  state.keys.clear(); state.padKey = ''; state.move = [0, 0]; updateControl();
  const pose = currentPose();
  lockMotionControls(true);
  kimodoProgress.hidden = false;
  kimodoPhase.textContent = 'Preparing entry blend…';
  kimodoProgressBar.value = 0;
  kimodoProgressLabel.textContent = '0%';
  statusElement.textContent = 'Loading Kimodo sequence…';
  try {
    const response = await api('/api/kimodo/start', {
      session: state.session, clip: kimodoSelect.value, advance: pose.frame,
      current_root: pose.root, current_rotations: pose.rotations,
    });
    state.motion = response.motion;
    state.targets = {frames: 0, joints: response.motion.joints, roots: [], rotations: []};
    state.playhead = 0;
    state.lastTime = performance.now();
    state.kimodo = {clip: response.clip, entryFrames: response.entry_frames, finishing: false, resume};
    setGroundPath(state.generatedPath, response.motion.roots, response.motion.frames);
    state.targetPath.visible = false;
    updateTargetVisibility();
    planInfo.textContent = `${response.clip.frames} authored frames · ${response.clip.name}`;
    statusElement.textContent = 'Kimodo sequence playing · controls locked';
    document.documentElement.dataset.kimodoState = 'playing';
    document.documentElement.dataset.kimodoClip = response.clip.id;
    document.documentElement.dataset.kimodoEntryFrames = String(response.entry_frames);
  } catch (error) {
    restoreKimodoAction(resume);
    kimodoProgress.hidden = true;
    lockMotionControls(false);
    statusElement.textContent = 'Kimodo start failed';
    throw error;
  }
}

function restoreKimodoAction(resume) {
  state.move = [...resume.move];
  state.facing = [...resume.facing];
  state.style = resume.style;
  state.padKey = resume.padKey;
  for (const button of document.querySelectorAll('.pad button')) {
    const pressed = button.dataset.key === state.padKey;
    button.classList.toggle('active', pressed);
    button.setAttribute('aria-pressed', String(pressed));
  }
}

async function finishKimodo() {
  if (!state.kimodo || state.kimodo.finishing) return;
  state.kimodo.finishing = true;
  kimodoPhase.textContent = 'Blending back to MotionBricks…';
  statusElement.textContent = 'Planning exit blend…';
  const resume = state.kimodo.resume;
  // Preserve the pre-clip action and its world-space movement/facing. The
  // authored clip may turn, but must not silently replace walking with idle.
  const facing = resume.facing;
  try {
    const response = await api('/api/kimodo/finish', {
      session: state.session, style: resume.style, move: resume.move, facing, seed: state.seed++,
    });
    state.kimodo = null;
    restoreKimodoAction(resume);
    useMotion(response, resume.move, 0, 0);
    kimodoProgress.hidden = true;
    lockMotionControls(false);
    document.documentElement.dataset.kimodoState = 'complete';
  } catch (error) {
    state.kimodo.finishing = false;
    statusElement.textContent = 'Kimodo exit failed · controls remain locked';
    document.documentElement.dataset.kimodoState = 'error';
    console.error(error);
  }
}

let controlTimer = 0;
function schedulePlan(delay = 0) { clearTimeout(controlTimer); controlTimer = setTimeout(() => void requestPlan(), delay); }
addEventListener('keydown', event => {
  if (state.controlsLocked) {
    if (['j', 'w', 'a', 's', 'd', 'arrowleft', 'arrowright', ' ', 'escape'].includes(event.key.toLowerCase())) event.preventDefault();
    return;
  }
  const key = event.key.toLowerCase();
  if (key === 'j' && !event.repeat) { beginJump(); event.preventDefault(); }
  if (['w', 'a', 's', 'd'].includes(key) && !state.keys.has(key)) {
    state.keys.add(key); updateControl(); schedulePlan(); event.preventDefault();
  }
  if (event.code === 'Space' || event.key === 'Escape') {
    state.keys.clear(); state.padKey = ''; updateControl(); schedulePlan(); event.preventDefault();
  }
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    const angle = Math.atan2(state.facing[0], state.facing[1]) + (event.key === 'ArrowLeft' ? 0.25 : -0.25);
    state.facing = [Math.sin(angle), Math.cos(angle)]; schedulePlan(35); event.preventDefault();
  }
});
jumpButton.addEventListener('click', beginJump);
kimodoPlay.addEventListener('click', () => { void beginKimodo(); });
jumpButton.setAttribute('aria-pressed', 'false');
addEventListener('keyup', event => {
  const key = event.key.toLowerCase();
  if (state.keys.delete(key)) { updateControl(); schedulePlan(); }
});
for (const button of document.querySelectorAll('.pad button')) {
  button.setAttribute('aria-pressed', 'false');
  button.addEventListener('click', event => {
    state.padKey = state.padKey === button.dataset.key ? '' : button.dataset.key;
    updateControl();
    schedulePlan();
    event.preventDefault();
  });
}
showAllTargets.addEventListener('change', updateTargetVisibility);
targetFrame.addEventListener('input', updateTargetVisibility);

function updateCamera() {
  // Use the pelvis, not the bounding box: waving an arm must not steer the
  // camera. Preserve this filter across plans and authored/live transitions.
  const followedRig=state.stream?.rendered?.physical ? state.stream.rig :
    (state.live?.enabled && state.live.rig ? state.live.rig : state.rig);
  cameraSubject.copy(followedRig.jointMeshes[0].position);
  cameraSubject.y -= 0.12;
  const now = performance.now();
  focus.fromArray(cameraFollow.update(cameraSubject.toArray(), (now - cameraFollowTime) / 1000));
  cameraFollowTime = now;
  const horizontal = Math.cos(cameraView.pitch) * cameraView.distance;
  desiredCamera.set(
    focus.x + Math.sin(cameraView.yaw) * horizontal,
    focus.y + Math.sin(cameraView.pitch) * cameraView.distance,
    focus.z + Math.cos(cameraView.yaw) * horizontal,
  );
  camera.position.copy(desiredCamera);
  camera.lookAt(focus);
}

function renderMotion(frame) {
  if (!state.motion) return;
  const index = Math.min(state.motion.frames - 1, Math.max(0, frame));
  if(state.live?.enabled) state.rig.poseInterpolated(state.motion.roots,state.motion.rotations,
    state.playhead+(state.live.queue.length?Math.min(1,state.live.elapsed/.02)*.6:0),state.motion.frames);
  else state.rig.pose(state.motion.roots, state.motion.rotations, index, state.motion.frames);
  updateCamera();
}

function installMotionQAHook() {
  if (query.get('qa') !== '1') return;
  window.__motionBricksQA = {
    pause(value = true) {
      state.qaPaused = value;
      state.lastTime = performance.now();
    },
    setFrame(frame) {
      if (!state.motion) return;
      state.playhead = THREE.MathUtils.clamp(Number(frame), 0, state.motion.frames - 1);
      renderMotion(Math.floor(state.playhead));
      renderer.render(scene, camera);
    },
    snapshot() {
      if (!state.motion) return null;
      const frame = THREE.MathUtils.clamp(Math.floor(state.playhead), 0, state.motion.frames - 1);
      if(!state.stream)renderMotion(frame);
      return {
        frame,
        root: state.motion.roots.slice(frame * 3, frame * 3 + 3),
        rotations: state.motion.rotations.slice(frame * state.motion.joints * 4, (frame + 1) * state.motion.joints * 4),
        joint_positions: state.rig.jointMeshes.flatMap(marker => marker.position.toArray()),
        motion_frames: state.motion.frames,
        joints: state.motion.joints,
        kimodo_state: document.documentElement.dataset.kimodoState || 'inactive',
        controls_locked: state.controlsLocked,
        entry_frames: state.kimodo?.entryFrames ?? 0,
        clip_frames: state.kimodo?.clip.frames ?? 0,
        clip_id: state.kimodo?.clip.id ?? '',
      };
    },
    motion() {
      if (!state.motion) return null;
      return {
        frames: state.motion.frames, joints: state.motion.joints,
        roots: state.motion.roots, rotations: state.motion.rotations,
        entry_frames: state.kimodo?.entryFrames ?? 0,
        clip_frames: state.kimodo?.clip.frames ?? 0,
      };
    },
  };
  document.documentElement.dataset.motionQA = 'available';
}

function replayRootPath(values, frames, joints, base = 0) {
  const roots = new Float32Array(frames * 3);
  for (let frame = 0; frame < frames; frame++) {
    const source = base + frame * joints * 3;
    const triple = values.subarray ? values.subarray(source, source + 3) : values.slice(source, source + 3);
    roots.set(triple, frame * 3);
  }
  return roots;
}

function buildComparisonShowcase(report) {
  const firstWalk = report.plans.find(plan => plan.style === 'walk' && Array.isArray(plan.movement));
  if (!firstWalk) return null;
  const heading = Math.atan2(firstWalk.movement[0], firstWalk.movement[2]);
  const angleFromHeading = plan => {
    if (!Array.isArray(plan.movement)) return 0;
    let difference = Math.atan2(plan.movement[0], plan.movement[2]) - heading;
    while (difference > Math.PI) difference -= 2 * Math.PI;
    while (difference < -Math.PI) difference += 2 * Math.PI;
    return Math.abs(difference);
  };
  const definitions = [
    {label: 'Forward walk', plans: report.plans.filter(plan => plan.style === 'walk' && angleFromHeading(plan) < Math.PI / 36)},
    {label: 'Right turn', plans: report.plans.filter(plan => plan.style === 'walk' && angleFromHeading(plan) >= Math.PI / 36)},
    {label: 'Zombie walk', plans: report.plans.filter(plan => plan.style === 'walk_zombie')},
  ];
  let cursor = 0;
  const segments = [];
  const slices = [];
  for (const definition of definitions) {
    if (!definition.plans.length) continue;
    const segmentStart = cursor;
    for (const plan of definition.plans) {
      const reportIndex = report.plans.indexOf(plan);
      const next = report.plans[reportIndex + 1];
      const available = Math.min(plan.expected_frames, plan.actual_frames);
      const untilNextPlan = next ? next.command_frame - plan.command_frame : available;
      const frames = THREE.MathUtils.clamp(untilNextPlan, 1, available);
      slices.push({start: cursor, frames, plan, label: definition.label});
      cursor += frames;
    }
    segments.push({label: definition.label, start: segmentStart, frames: cursor - segmentStart});
  }
  return segments.length === 3 ? {frames: cursor, segments, slices} : null;
}

function comparisonPosition(frame) {
  if (state.comparisonPlan >= 0) {
    const plan = state.comparison.plans[state.comparisonPlan];
    const frames = Math.min(plan.expected_frames, plan.actual_frames);
    return {plan, index: THREE.MathUtils.clamp(frame, 0, frames - 1), frames,
      label: null, sequenceFrame: frame};
  }
  const showcase = state.comparisonShowcase;
  const sequenceFrame = THREE.MathUtils.clamp(frame, 0, showcase.frames - 1);
  const slice = showcase.slices.find(item => sequenceFrame < item.start + item.frames) ?? showcase.slices.at(-1);
  return {plan: slice.plan, index: sequenceFrame - slice.start, frames: showcase.frames,
    label: slice.label, sequenceFrame};
}

function setComparisonPaths(plan) {
  if (state.comparisonVisiblePlan === plan.index) return;
  state.comparisonVisiblePlan = plan.index;
  setGroundPath(state.generatedPath,
    replayRootPath(plan.expected_joint_positions, plan.expected_frames, state.comparison.joints), plan.expected_frames);
  setGroundPath(state.targetPath,
    replayRootPath(plan.native_joint_positions, plan.actual_frames, state.comparison.joints), plan.actual_frames);
}

function renderComparison(frame) {
  const report = state.comparison;
  if (!report) return;
  const position = comparisonPosition(frame);
  const {plan, index} = position;
  setComparisonPaths(plan);
  const width = report.joints * 3;
  const expectedOffset = index * width;
  const nativeOffset = index * width;
  state.rig.poseWorld(plan.expected_joint_positions, expectedOffset);
  state.nativeRig.poseWorld(plan.native_joint_positions, nativeOffset);
  const positions = state.errorLines.geometry.attributes.position.array;
  let maximum = 0, square = 0;
  for (let joint = 0; joint < report.joints; joint++) {
    const expected = expectedOffset + joint * 3;
    const native = nativeOffset + joint * 3;
    const line = joint * 6;
    let norm = 0;
    for (let axis = 0; axis < 3; axis++) {
      positions[line + axis] = plan.expected_joint_positions[expected + axis];
      positions[line + 3 + axis] = plan.native_joint_positions[native + axis];
      const difference = positions[line + 3 + axis] - positions[line + axis];
      norm += difference * difference;
    }
    maximum = Math.max(maximum, Math.sqrt(norm));
    square += norm;
  }
  state.errorLines.geometry.attributes.position.needsUpdate = true;
  const rootMetric = plan.metrics.root_m.max;
  planInfo.textContent = position.label
    ? `${position.label} · sequence ${position.sequenceFrame}/${position.frames - 1} · plan ${plan.index} frame ${index}`
    : `plan ${plan.index} · ${plan.style.replaceAll('_', ' ')} · frame ${index}/${position.frames - 1}`;
  targetInfo.textContent = `frame max ${formatDistance(maximum)} · RMS ${formatDistance(Math.sqrt(square / report.joints))}`;
  statusElement.textContent = `${plan.duration_exact ? 'duration exact' : 'duration mismatch'} · root max ${formatDistance(rootMetric)}`;
  replayFrame.value = String(position.sequenceFrame);
  replayFrameLabel.textContent = `${position.sequenceFrame}/${position.frames - 1}`;
  updateCamera();
}

function formatDistance(metres) {
  return metres < 0.01 ? `${(metres * 1000).toFixed(1)} mm` : `${(metres * 100).toFixed(1)} cm`;
}

function renderReplay(frame) {
  const replay = state.replay;
  if (!replay) return;
  const index = THREE.MathUtils.clamp(frame, 0, replay.frames - 1);
  state.rig.poseWorld(replay.jointPositions, index * replay.joints * 3);
  const plan = replay.framePlans[index];
  if (plan !== state.replayPlan) {
    state.replayPlan = plan;
    if (plan >= 0) {
      const planBase = plan * replay.targetFrames * replay.joints * 3;
      for (let target = 0; target < replay.targetFrames; target++) {
        state.targetRigs[target].poseWorld(replay.targetPositions, planBase + target * replay.joints * 3);
      }
      setGroundPath(state.targetPath, replayRootPath(replay.targetPositions, replay.targetFrames, replay.joints, planBase), replay.targetFrames);
      state.targets = {frames: replay.targetFrames};
    } else {
      state.targets = {frames: 0};
    }
    updateTargetVisibility();
  }
  const mode = modeNames[replay.modes[index]] || `mode ${replay.modes[index]}`;
  planInfo.textContent = plan >= 0
    ? `frame ${index}/${replay.frames - 1} · plan ${plan} · ${mode.replaceAll('_', ' ')}`
    : `frame ${index}/${replay.frames - 1} · pre-roll`;
  replayFrame.value = String(index);
  replayFrameLabel.textContent = `${index}/${replay.frames - 1}`;
  updateCamera();
}

function animate(now) {
  requestAnimationFrame(animate);
  const delta = state.qaPaused ? 0 : Math.min(0.1, (now - state.lastTime) / 1000);
  state.lastTime = now;
  if (state.stream) {
    renderStream(delta);
  } else if (state.physics) {
    if (state.replayPlaying) {
      state.physicsTime = Math.min(state.physics.times.at(-1), state.physicsTime + delta);
      if (state.physicsTime === state.physics.times.at(-1)) {
        state.replayPlaying = false;
        replayToggle.textContent = 'Replay';
      }
    }
    renderPhysics();
  } else if (state.comparison) {
    const frames = state.comparisonPlan < 0 ? state.comparisonShowcase.frames
      : Math.min(state.comparison.plans[state.comparisonPlan].expected_frames,
        state.comparison.plans[state.comparisonPlan].actual_frames);
    if (state.replayPlaying) state.playhead = (state.playhead + delta * state.comparison.fps) % frames;
    renderComparison(Math.floor(state.playhead));
  } else if (state.replay) {
    if (state.replayPlaying) state.playhead = (state.playhead + delta * state.replay.fps) % state.replay.frames;
    renderReplay(Math.floor(state.playhead));
  } else if (state.motion) {
    if (state.live?.enabled) advanceLivePhysics(delta);
    else state.playhead += delta * state.meta.fps;
    if (state.kimodo) {
      const authoredFrame = state.playhead - state.kimodo.entryFrames;
      const progress = THREE.MathUtils.clamp(authoredFrame / Math.max(1, state.kimodo.clip.frames - 1), 0, 1);
      kimodoProgressBar.value = progress;
      kimodoProgressLabel.textContent = `${Math.round(progress * 100)}%`;
      kimodoPhase.textContent = authoredFrame < 0 ? 'Blending into Kimodo…'
        : (state.kimodo.finishing ? 'Blending back to MotionBricks…' : 'Kimodo sequence playing');
      document.documentElement.dataset.kimodoProgress = String(progress);
      if (state.playhead >= state.motion.frames - 1 && !(state.live?.busy || state.live?.queue.length)) {
        state.playhead = state.motion.frames - 1;
        void finishKimodo();
      }
    } else {
      const moving = Math.hypot(state.move[0], state.move[1]) > 1e-6;
      if (state.jumpActive && state.playhead >= state.motion.frames - 1 && !state.pending)
        void requestPlan(state.motion.frames - 1);
      else if (!state.jumpActive && moving && state.playhead >= controllerCadenceFrames && !state.pending)
        void requestPlan(controllerCadenceFrames);
    }
    renderMotion(Math.floor(state.playhead));
    if (state.live?.enabled) renderLivePhysics();
  }
  renderer.render(scene, camera);
}

replayToggle.addEventListener('click', () => {
  if (state.physics && state.physicsTime >= state.physics.times.at(-1)) {
    state.physicsTime = 0;
    cameraFollow.position = null; // Explicit replay jump, not a motion transition.
  }
  state.replayPlaying = !state.replayPlaying;
  replayToggle.textContent = state.replayPlaying ? 'Pause' : 'Play';
  state.lastTime = performance.now();
});
replayFrame.addEventListener('input', () => {
  state.replayPlaying = false;
  replayToggle.textContent = 'Play';
  state.playhead = Number(replayFrame.value);
  if (state.physics) {
    state.physicsTime = state.physics.times[state.playhead];
    cameraFollow.position = null; // Scrubbing must immediately frame the selected pose.
    renderPhysics();
  } else if (state.comparison) renderComparison(Math.floor(state.playhead));
  else renderReplay(Math.floor(state.playhead));
});

comparisonPlan.addEventListener('change', () => {
  state.comparisonPlan = Number(comparisonPlan.value);
  state.playhead = 0;
  const frames = state.comparisonPlan < 0 ? state.comparisonShowcase.frames
    : Math.min(state.comparison.plans[state.comparisonPlan].expected_frames,
      state.comparison.plans[state.comparisonPlan].actual_frames);
  replayFrame.max = String(frames - 1);
  state.comparisonVisiblePlan = -1;
  renderComparison(0);
});

async function selfTest() {
  const alternate = state.meta.styles.find(item => item.name === 'walk_zombie') || state.meta.styles.find(item => item.name !== 'walk');
  if (!alternate) throw new Error('no alternate upstream style available');
  state.style = alternate.name;
  styleSelect.value = alternate.name;
  state.move = [1, 0];
  state.facing = [1, 0];
  const response = await requestPlan(3);
  if (!response || response.style !== alternate.name) throw new Error('style change was not applied');
  if (response.motion.joints !== 34 || response.motion.frames < 24 || response.motion.rotations.length !== response.motion.frames * 34 * 4) {
    throw new Error('invalid skeletal animation response');
  }
  if (response.targets.frames !== 4 || response.targets.joints !== 34 || response.targets.roots.length !== 12 || response.targets.rotations.length !== 4 * 34 * 4) {
    throw new Error('invalid placed target-keyframe response');
  }
  renderMotion(2);
  renderer.render(scene, camera);
  let visibleTargets = state.targetRigs.filter(rig => rig.group.visible).length;
  if (!renderer.domElement.width || state.rig.bones.length !== 34 || visibleTargets !== 1 || !state.targetRigs[3].group.visible) {
    throw new Error('animated and target skeletons were not rendered');
  }
  const expectedFocus = state.rig.jointMeshes[0].position.clone();
  expectedFocus.y -= 0.12;
  if (cameraSubject.distanceTo(expectedFocus) > 1e-6 || !focus.toArray().every(Number.isFinite))
    throw new Error('camera subject/filter is invalid');
  targetFrame.value = '1';
  updateTargetVisibility();
  if (!state.targetRigs[1].group.visible || state.targetRigs.filter(rig => rig.group.visible).length !== 1) {
    throw new Error('target-frame inspector did not select T1');
  }
  showAllTargets.checked = true;
  updateTargetVisibility();
  visibleTargets = state.targetRigs.filter(rig => rig.group.visible).length;
  if (visibleTargets !== 4) throw new Error('four-pose target overlay did not render');
  showAllTargets.checked = false;
  targetFrame.value = '3';
  updateTargetVisibility();
  document.documentElement.dataset.animatedJoints = String(state.rig.bones.length);
  document.documentElement.dataset.targetFrames = String(response.targets.frames);
  document.documentElement.dataset.visibleTargets = '1';
  document.documentElement.dataset.cameraSubject = 'animated';
  document.documentElement.dataset.testStatus = 'passed';
  testResult.textContent = `Headless check passed: animated-camera anchor + target inspector/overlay, ${alternate.name}, right turn`;
}

async function replaySelfTest() {
  const replay = state.replay;
  state.replayPlaying = false;
  replayToggle.textContent = 'Play';
  const plan = Math.min(7, replay.plans - 1);
  state.playhead = replay.planFrames[plan];
  renderReplay(Math.floor(state.playhead));
  showAllTargets.checked = true;
  updateTargetVisibility();
  renderer.render(scene, camera);
  const visibleTargets = state.targetRigs.filter(rig => rig.group.visible).length;
  const expectedFocus = state.rig.jointMeshes[0].position.clone();
  expectedFocus.y -= 0.12;
  if (replay.frames !== state.meta.frames || replay.joints !== state.meta.joints ||
      replay.plans !== state.meta.plans || replay.targetFrames !== 4 || visibleTargets !== 4 ||
      cameraSubject.distanceTo(expectedFocus) > 1e-6 || !focus.toArray().every(Number.isFinite) || !renderer.domElement.width) {
    throw new Error('replay dimensions, targets, or animated-camera anchor are invalid');
  }
  document.documentElement.dataset.replayFrames = String(replay.frames);
  document.documentElement.dataset.animatedJoints = String(replay.joints);
  document.documentElement.dataset.replayPlans = String(replay.plans);
  document.documentElement.dataset.targetFrames = String(replay.targetFrames);
  document.documentElement.dataset.visibleTargets = String(visibleTargets);
  document.documentElement.dataset.cameraSubject = 'animated';
  document.documentElement.dataset.runtime = 'replay';
  document.documentElement.dataset.testStatus = 'passed';
  testResult.textContent = `Replay check passed: ${replay.frames} frames, ${replay.plans} plans, four target ghosts, animated-camera anchor`;
}

async function startReplay() {
  state.replay = await loadReplay();
  if (state.replay.fps !== state.meta.fps || state.replay.frames !== state.meta.frames ||
      state.replay.joints !== state.meta.joints || state.replay.plans !== state.meta.plans) {
    throw new Error('replay metadata does not match its payload');
  }
  const joints = Array.from({length: state.replay.joints}, (_, index) => ({
    name: `joint-${index}`,
    parent: state.replay.parents[index],
    position: Array.from(state.replay.jointPositions.subarray(index * 3, index * 3 + 3)),
  }));
  makeSkeletons(joints);
  setGroundPath(state.generatedPath, replayRootPath(state.replay.jointPositions, state.replay.frames, state.replay.joints), state.replay.frames);
  document.querySelector('#live-style').hidden = true;
  document.querySelector('.pad').hidden = true;
  document.querySelector('.live-actions').hidden = true;
  replayControls.hidden = false;
  replayFrame.max = String(state.replay.frames - 1);
  document.querySelector('#lede').textContent = 'Deterministic replay of the captured upstream session. Scrub or play without invoking either planner.';
  document.querySelector('#camera-help').textContent = 'Drag to orbit · wheel to zoom · the camera follows only the animated skeleton';
  document.querySelector('#backend').textContent = 'captured artifact';
  statusElement.textContent = 'Replaying';
  showAllTargets.checked = true;
  renderReplay(0);
  requestAnimationFrame(animate);
  if (query.get('test') === '1') await replaySelfTest();
  else document.documentElement.dataset.testStatus = 'ready';
}

async function comparisonSelfTest() {
  if (!state.comparisonShowcase || state.comparisonShowcase.segments.length !== 3 ||
      state.comparisonPlan !== -1 || Number(replayFrame.max) !== state.comparisonShowcase.frames - 1 ||
      state.meta.passed !== true || state.comparison.passed !== true) {
    throw new Error('comparison showcase is unavailable or is not the default');
  }
  const labels = state.comparisonShowcase.segments.map(segment => segment.label).join(',');
  for (const segment of state.comparisonShowcase.segments) renderComparison(segment.start);
  state.replayPlaying = false;
  replayToggle.textContent = 'Play';
  state.comparisonPlan = Math.min(7, state.comparison.plans.length - 1);
  comparisonPlan.value = String(state.comparisonPlan);
  comparisonPlan.dispatchEvent(new Event('change'));
  state.playhead = Math.min(8, Number(replayFrame.max));
  renderComparison(state.playhead);
  renderer.render(scene, camera);
  if (state.comparison.plans.length !== state.meta.plans || state.comparison.joints !== 34 ||
      state.rig.jointMeshes.length !== 34 || state.nativeRig.jointMeshes.length !== 34 ||
      state.errorLines.geometry.attributes.position.count !== 68 || !renderer.domElement.width) {
    throw new Error('comparison plan, overlay rigs, or error vectors are invalid');
  }
  document.documentElement.dataset.runtime = 'comparison';
  document.documentElement.dataset.comparisonPlans = String(state.comparison.plans.length);
  document.documentElement.dataset.animatedJoints = String(state.comparison.joints);
  document.documentElement.dataset.errorVectors = String(state.comparison.joints);
  document.documentElement.dataset.comparisonShowcase = labels;
  document.documentElement.dataset.comparisonShowcaseFrames = String(state.comparisonShowcase.frames);
  document.documentElement.dataset.comparisonPassed = String(state.comparison.passed);
  document.documentElement.dataset.testStatus = 'passed';
  testResult.textContent = `Parity viewer check passed: ${state.comparison.plans.length} independent plans, upstream/native overlay, ${state.comparison.joints} error vectors`;
}

async function startComparison() {
  state.comparison = await api('/api/comparison');
  if (state.comparison.format !== 'motionbricks-open-loop-report-v1' ||
      state.comparison.fps !== state.meta.fps || state.comparison.joints !== state.meta.joints ||
      state.comparison.plans.length !== state.meta.plans || state.comparison.passed !== state.meta.passed) {
    throw new Error('comparison metadata does not match its report');
  }
  document.documentElement.dataset.comparisonPassed = String(state.comparison.passed);
  const joints = Array.from({length: state.comparison.joints}, (_, index) => ({
    name: `joint-${index}`, parent: state.comparison.parents[index],
    position: state.comparison.neutral_joints.slice(index * 3, index * 3 + 3),
  }));
  makeComparisonSkeletons(joints);
  state.comparisonShowcase = buildComparisonShowcase(state.comparison);
  if (!state.comparisonShowcase) throw new Error('comparison report does not contain the three-motion showcase');
  const showcaseOption = document.createElement('option');
  showcaseOption.value = '-1';
  showcaseOption.textContent = 'Showcase · forward walk → right turn → zombie walk';
  comparisonPlan.append(showcaseOption);
  for (const plan of state.comparison.plans) {
    const option = document.createElement('option');
    option.value = String(plan.index);
    option.textContent = `plan ${plan.index} · ${plan.style.replaceAll('_', ' ')} · ${plan.actual_frames}/${plan.expected_frames} frames`;
    comparisonPlan.append(option);
  }
  document.querySelector('#live-style').hidden = true;
  document.querySelector('.pad').hidden = true;
  document.querySelector('.live-actions').hidden = true;
  document.querySelector('.target-frame-control').hidden = true;
  document.querySelector('.view-options label').hidden = true;
  document.querySelector('.target-help').hidden = true;
  replayControls.hidden = false;
  comparisonPlanRow.hidden = false;
  document.querySelector('label[for="replay-frame"]').childNodes[0].textContent = 'Comparison frame ';
  document.querySelector('.legend span:first-child').lastChild.textContent = 'Upstream';
  document.querySelector('.legend span:last-child').lastChild.textContent = 'Native';
  document.querySelector('.target-swatch').style.background = '#65a9ff';
  document.querySelector('.target-swatch').style.color = '#65a9ff';
  const verdict = state.comparison.passed ? 'Strict parity passed.' : 'Strict parity failed.';
  document.querySelector('#lede').textContent = `${verdict} Open-loop comparison of forward walking, a right turn, and zombie walking. Each replan starts from the same recorded upstream context.`;
  document.querySelector('#camera-help').textContent = 'Drag to orbit · wheel to zoom · solid green is upstream, blue diamonds are native';
  document.querySelector('#backend').textContent = `${state.comparison.device} parity report`;
  comparisonPlan.value = '-1';
  comparisonPlan.dispatchEvent(new Event('change'));
  requestAnimationFrame(animate);
  if (query.get('test') === '1') await comparisonSelfTest();
  else document.documentElement.dataset.testStatus = 'ready';
}

function renderPhysics() {
  const p = state.physics, stride = p.joints.length * 3;
  let index = 0;
  while (index + 1 < p.frames && p.times[index + 1] <= state.physicsTime) index++;
  const next = Math.min(index + 1, p.frames - 1);
  const blend = next === index ? 0 : (state.physicsTime - p.times[index]) / (p.times[next] - p.times[index]);
  // Interpolate both recorded skeletons on the same clock, avoiding 50 Hz
  // camera/pose sample-and-hold jitter at higher browser refresh rates.
  for (let j = 0; j < stride; j++) {
    p.actualPose[j] = THREE.MathUtils.lerp(p.actual_positions[index*stride+j],p.actual_positions[next*stride+j],blend);
    p.referencePose[j] = THREE.MathUtils.lerp(p.reference_positions[index*stride+j],p.reference_positions[next*stride+j],blend);
  }
  state.rig.poseWorld(p.actualPose);
  state.nativeRig.poseWorld(p.referencePose);
  const lines = state.errorLines.geometry.attributes.position;
  for (let j = 0; j < p.joints.length; j++) {
    lines.setXYZ(j*2,...p.actualPose.subarray(j*3,j*3+3));
    lines.setXYZ(j*2+1,...p.referencePose.subarray(j*3,j*3+3));
  }
  lines.needsUpdate = true;
  state.errorLines.geometry.computeBoundingSphere();
  replayFrame.value = String(index);
  replayFrameLabel.textContent = `${state.physicsTime.toFixed(2)} / ${p.times.at(-1).toFixed(2)} s`;
  planInfo.textContent = `Root drift ${p.root_error_m[index].toFixed(2)} m · mean body error ${p.body_error_m[index].toFixed(2)} m`;
  targetInfo.textContent = `${p.contacts[index]} contact points · reference aligned only at start`;
  document.documentElement.dataset.physicsFrame = String(index);
  document.documentElement.dataset.physicsRootError = String(p.root_error_m[index]);
  updateCamera();
}

async function installPhysics() {
  const response = await fetch('/api/physics');
  if (response.status === 404) return false; // Older/replay-only handlers.
  if (!response.ok) throw new Error(`Physics index: HTTP ${response.status}`);
  const recordings = await response.json();
  const select = document.querySelector('#physics-select');
  document.querySelector('#physics-controls').hidden = recordings.length === 0;
  for (const recording of recordings) {
    const option = document.createElement('option');
    option.value = recording.id; option.textContent = recording.title;
    select.append(option);
  }
  select.addEventListener('change', () => {
    const url = new URL(location.href);
    if (select.value) url.searchParams.set('physics',select.value);
    else url.searchParams.delete('physics');
    location.assign(url);
  });
  const id = query.get('physics');
  if (!id) return false;
  if (!recordings.some(r => r.id === id)) throw new Error('Unknown physical recording');
  select.value = id;
  const p = state.physics = await api(`/api/physics/${encodeURIComponent(id)}`);
  state.controlsLocked = true;
  cameraView.distance = 6.5;
  p.actualPose = new Float32Array(p.joints.length * 3);
  p.referencePose = new Float32Array(p.joints.length * 3);
  makeComparisonSkeletons(p.joints,'PHYSICAL','REFERENCE');
  setGroundPath(state.generatedPath,replayRootPath(p.actual_positions,p.frames,p.joints.length),p.frames);
  setGroundPath(state.targetPath,replayRootPath(p.reference_positions,p.frames,p.joints.length),p.frames);
  for (const selector of ['#live-style','#kimodo-controls','.pad','.live-actions','.target-frame-control','.view-options label','.target-help'])
    document.querySelector(selector).hidden = true;
  replayControls.hidden = false;
  replayFrame.max = String(p.frames-1);
  document.querySelector('#physics-overlay-label').hidden = false;
  document.querySelector('#physics-overlay').addEventListener('change', event => {
    state.nativeRig.setVisible(event.target.checked);
    state.errorLines.visible = event.target.checked;
    state.targetPath.visible = event.target.checked;
  });
  document.querySelector('.legend span:first-child').lastChild.textContent = 'Physical robot';
  document.querySelector('.legend span:last-child').lastChild.textContent = 'Reference motion';
  document.querySelector('.target-swatch').style.background = '#65a9ff';
  document.querySelector('#lede').textContent = 'Recorded SONIC / MuJoCo simulation, not live physics. Green is the actuated robot; blue is the reference. Pink links show tracking error, including trajectory drift.';
  document.querySelector('#camera-help').textContent = 'Drag to orbit · wheel to zoom · scrub to inspect · camera follows the physical robot';
  document.querySelector('#backend').textContent = 'Upstream SONIC · TensorRT · MuJoCo';
  statusElement.textContent = p.diagnostics.failure
    ? `Recorded simulation · FAILED: ${p.diagnostics.failure}`
    : 'Recorded simulation · no falls or resets';
  testResult.textContent = `Joint RMSE ${p.diagnostics.joint_rmse_rad.toFixed(3)} rad · final root drift ${p.diagnostics.root_xy_final_error_m.toFixed(2)} m`;
  if (p.diagnostics.contact_slip)
    testResult.textContent += ` · mean contact slip ${(p.diagnostics.contact_slip.tangential_speed_mean_mps*100).toFixed(1)} cm/s`;
  renderPhysics();
  requestAnimationFrame(animate);
  document.documentElement.dataset.testStatus = 'ready';
  return true;
}

async function installStream() {
  const response=await fetch('/api/stream');if(response.status===404)return false;
  if(!response.ok)throw new Error('Cannot query streaming capabilities');
  const capability=await response.json();if(!capability.available)return false;
  const section=document.querySelector('#live-physics-controls'),toggle=document.querySelector('#live-physics');
  const reset=document.querySelector('#live-physics-reset'),pause=document.querySelector('#stream-pause');
  const collisionToggle=document.querySelector('#show-collisions');
  collisionToggle.disabled=!capability.physics;
  section.hidden=false;toggle.disabled=!capability.physics;reset.textContent='Reset session';pause.hidden=false;
  state.session='server-stream';state.targets={frames:0,joints:34,roots:[],rotations:[]};
  const s=state.stream={client:null,rig:null,label:null,paused:false,physics:false,owner:false,rendered:null,lastEpoch:null,error:''};
  const info=document.querySelector('#live-physics-status');
  const prepareCollisions=()=>{
    if(!s.collisionDefinitions||s.collisions)return;
    s.collisions=new CollisionGeometry(s.collisionDefinitions.shapes);scene.add(s.collisions.group);
    s.collisionsID=s.collisionDefinitions.id;
  };
  const client=s.client=new StreamClient(message=>{
    if(message.type==='hello') {
      s.owner=message.owner;lockMotionControls(!s.owner);reset.disabled=!s.owner;pause.disabled=!s.owner;toggle.disabled=!s.owner||!capability.physics;
      statusElement.textContent=s.owner?'Connected · server controls motion and simulation':'Spectating · another client controls the session';
      document.documentElement.dataset.streamConnected='true';return;
    }
    if(message.type==='disconnected') {lockMotionControls(true);toggle.disabled=true;reset.disabled=true;pause.disabled=true;statusElement.textContent='Disconnected · reconnecting without resetting the session';document.documentElement.dataset.streamConnected='false';return;}
    if(message.type==='error') {s.error=message.error;info.textContent=message.error;statusElement.textContent=`Simulation stopped: ${message.error}`;document.documentElement.dataset.streamError=message.error;return;}
    if(message.type==='ack') {document.documentElement.dataset.streamAck=JSON.stringify(message);if(message.state==='rejected')info.textContent=message.error;return;}
    if(message.type==='targets') {
      state.targets=message.targets??{frames:0,joints:34,roots:[],rotations:[]};
      for(let i=0;i<state.targets.frames;i++)state.targetRigs[i].pose(state.targets.roots,state.targets.rotations,i,state.targets.frames);
      document.documentElement.dataset.planSequence=String(Number(document.documentElement.dataset.planSequence||0)+1);
      updateTargetVisibility();return;
    }
    if(message.type==='collision_shapes') {
      if(message.id&&s.collisionsID===message.id&&s.collisions)return;
      s.collisions?.dispose();s.collisions=null;s.collisionDefinitions=message;
      if(collisionToggle.checked)prepareCollisions();
      document.documentElement.dataset.collisionShapeCount=String(message.shapes.length);return;
    }
    if(message.type==='frame') {
      document.documentElement.dataset.livePhysicsTime=String(message.time);
      document.documentElement.dataset.streamTick=String(message.tick);
      document.documentElement.dataset.streamError=message.error??'';
      document.documentElement.dataset.streamHolding=String(message.holding===true);
      document.documentElement.dataset.streamFallen=String(message.fallen===true);
      document.documentElement.dataset.streamPhysicsTime=String(message.physics_time??0);
      s.planning=message.planning===true;
      s.error=message.error??'';
      const uiKey=`${message.epoch}:${message.physics}:${message.paused}:${message.kind}:${message.style}:${message.error}:${message.fallen}`;
      if(s.uiKey===uiKey&&performance.now()-(s.uiAt??0)<100)return;
      s.uiKey=uiKey;s.uiAt=performance.now();
      if(message.epoch!==s.lastEpoch){s.lastEpoch=message.epoch;cameraFollow.position=null;}
      s.paused=message.paused;pause.textContent=s.paused?'Resume':'Pause';s.physics=message.physics;
      pause.disabled=!s.owner||Boolean(message.error);
      toggle.checked=s.physics;document.documentElement.dataset.livePhysics=String(s.physics);
      const locked=message.kind!=='motion';lockMotionControls(!s.owner||locked);
      state.jumpActive=message.kind==='jump';
      document.documentElement.dataset.kimodoState=message.kind==='kimodo'?'playing':'inactive';
      document.documentElement.dataset.plannedJump=String(state.jumpActive);
      kimodoProgress.hidden=message.kind!=='kimodo';kimodoProgressBar.value=message.progress;kimodoProgressLabel.textContent=`${Math.round(message.progress*100)}%`;kimodoPhase.textContent='Server-controlled Kimodo playback';
      planInfo.textContent=`${message.style.replaceAll('_',' ')} · server time ${message.time.toFixed(2)} s`;
      state.rig.boneMaterial.color.setHex(s.physics?0x65a9ff:0x55efc4);state.rig.jointMaterial.color.setHex(s.physics?0xaacaff:0xd9fff3);state.rig.setOpacity(s.physics?.65:1);
      document.querySelector('.legend span:first-child').lastChild.textContent=s.physics?'Generated reference':'Animated model';
      if(message.error)info.textContent=`${message.error}. Disable Live physics to continue the reference animation, or Reset session to start over.`;
      else info.textContent=`WebSocket · ${s.physics?'GGML SONIC + MuJoCo':'kinematic reference'} · ${message.paused?'paused':'50 Hz server stream'} · ${client.buffer.underruns} playback underruns${message.fallen?' · Fall detected; physics continues':''}`;
    }
  });
  toggle.addEventListener('change',()=>client.send('physics',{enabled:toggle.checked}));
  collisionToggle.addEventListener('change',()=>{if(collisionToggle.checked){try{prepareCollisions()}catch(error){collisionToggle.checked=false;info.textContent=`Collision overlay unavailable: ${error.message}`;}}});
  reset.addEventListener('click',()=>client.send('reset'));
  pause.addEventListener('click',()=>client.send(s.paused?'resume':'pause'));
  if(query.get('qa')==='1')window.__motionBricksStreamQA={
    snapshot:()=>({rendered:s.rendered,queue:client.buffer.frames.length,underruns:client.buffer.underruns,waiting:client.buffer.waiting,serverTime:client.buffer.frames.at(-1)?.time??0}),
    send:(type,values)=>client.send(type,values),disconnect:()=>client.socket.close(),
  };
  return true;
}

function renderStream(delta) {
  const s=state.stream,sample=s.client.buffer.sample(delta);if(!sample)return;
  const {a,b,alpha}=sample;
  const roots=[...a.root,...b.root],rotations=[...a.rotations,...b.rotations];
  state.motion={frames:2,joints:34,roots,rotations};state.playhead=alpha;
  state.rig.poseInterpolated(roots,rotations,alpha,2);
  const root=a.root.map((v,i)=>v+(b.root[i]-v)*alpha);
  let physical=null;
  if(a.physics&&b.physics&&a.physical&&b.physical) {
    if(!s.rig){
      const joints=a.parents.map((parent,i)=>({parent,name:`physical_${i}`,position:a.physical.slice(i*3,i*3+3)}));
      s.rig=new SkeletonRig(joints,{color:0x55efc4,jointColor:0xd9fff3,rootColor:0xffd166,emissive:0x0c5b49,rootEmissive:0x6a3b00,emissiveIntensity:.8,radius:.022,jointRadius:.034,rootRadius:.062,opacity:1,diamonds:false,renderOrder:7,label:'PHYSICAL',labelColor:'#55efc4'});
      s.label=labelSprite('REFERENCE','#65a9ff',.9);scene.add(s.label);
    }
    physical=a.physical.map((v,i)=>v+(b.physical[i]-v)*alpha);s.rig.poseWorld(physical);s.rig.setVisible(true);s.label.visible=true;s.label.position.copy(state.rig.jointMeshes[0].position).add(new THREE.Vector3(-.25,.5,0));
  }else if(s.rig){s.rig.setVisible(false);s.label.visible=false;}
  if(s.collisions) {
    if(physical&&document.querySelector('#show-collisions').checked)s.collisions.pose(a.collision_transforms,b.collision_transforms,alpha);
    else s.collisions.group.visible=false;
    document.documentElement.dataset.collisionsVisible=String(s.collisions.group.visible);
  }
  s.rendered={time:a.time+(b.time-a.time)*alpha,root,physical,epoch:a.epoch,buffering:sample.buffering,holding:a.holding===true&&b.holding===true};
  if(query.get('qa')==='1')s.rendered.reference=state.rig.jointMeshes.flatMap(joint=>joint.position.toArray());
  if(query.get('qa')==='1'&&s.collisions?.group.visible)s.rendered.collisions=s.collisions.meshes.map(mesh=>({name:mesh.name,position:mesh.position.toArray(),rotation:mesh.quaternion.toArray(),opacity:mesh.material.opacity}));
  document.documentElement.dataset.streamPlaybackTime=String(s.rendered.time);
  document.documentElement.dataset.streamUnderruns=String(s.client.buffer.underruns);
  if(s.client.socket.readyState===WebSocket.OPEN)statusElement.textContent=s.error?`Simulation stopped: ${s.error}`:s.paused?'Simulation paused':s.planning?'Waiting for motion planner…':sample.buffering?'Buffering streamed poses…':'Streaming · interpolated playback';
  updateCamera();
}

async function installLivePhysics() {
  const response = await fetch('/api/live-physics');
  if (response.status === 404) return;
  if (!response.ok) throw new Error('Cannot query live physics');
  const info = await response.json();
  if (!info.available) return;
  const section = document.querySelector('#live-physics-controls');
  const toggle = document.querySelector('#live-physics');
  const reset = document.querySelector('#live-physics-reset');
  section.hidden = false;
  state.live = {enabled:false, busy:false, queue:[], elapsed:0, rig:null, current:null,
    previous:null, fallen:false, deferredPlan:null, deferredKimodo:false,
    owner:Array.from(crypto.getRandomValues(new Uint32Array(4))).join('-')};
  async function release() {
    const l=state.live;
    if(l.busy) return false;
    l.busy=true;
    try {
      await api('/api/live-physics',{session:l.owner,release:true});
      l.queue=[];l.current=null;l.previous=null;l.elapsed=0;l.fallen=false;
      l.deferredPlan=null;l.deferredKimodo=false;
      document.querySelector('#live-physics-status').textContent='Physical state reset explicitly. Waiting for the next control tick.';
      return true;
    } finally {l.busy=false;}
  }
  toggle.addEventListener('change', async () => {
    const l=state.live;
    if(l.busy || l.queue.length) {l.disableRequested=!toggle.checked;toggle.checked=l.enabled;return;}
    try {
      if(!toggle.checked) await release();
      l.enabled=toggle.checked;reset.disabled=!l.enabled;
      if(l.rig) l.rig.setVisible(l.enabled);
      if(l.referenceLabel) l.referenceLabel.visible=l.enabled;
      state.rig.boneMaterial.color.setHex(l.enabled?0x65a9ff:0x55efc4);
      state.rig.jointMaterial.color.setHex(l.enabled?0xaacaff:0xd9fff3);
      state.rig.setOpacity(l.enabled?.65:1);
      document.querySelector('.legend span:first-child').lastChild.textContent=l.enabled?'Generated reference':'Animated model';
      document.querySelector('.animated-swatch').style.background=l.enabled?'#65a9ff':'#55efc4';
      document.documentElement.dataset.livePhysics=String(l.enabled);
      document.querySelector('#live-physics-status').textContent=l.enabled
        ? 'Live GGML SONIC + MuJoCo · blue reference, green physical, warm target keyframes. No pose resets on replanning or Kimodo transitions.'
        : 'Physics off · kinematic playback';
    } catch(error) {document.querySelector('#live-physics-status').textContent=error.message;}
  });
  reset.addEventListener('click',async()=>{
    if(state.live.busy || state.live.queue.length) {state.live.resetRequested=true;return;}
    try {await release();} catch(error){document.querySelector('#live-physics-status').textContent=error.message;}
  });
}

async function requestLiveBatch() {
  const l=state.live;
  if(!l?.enabled || l.busy || l.queue.length || l.fallen || state.pending ||
      (state.controlsLocked && !state.kimodo) || state.kimodo?.finishing) return;
  if(l.deferredKimodo) {l.deferredKimodo=false;void beginKimodo();return;}
  if(l.deferredPlan && !state.kimodo) {
    const request=l.deferredPlan;l.deferredPlan=null;void requestPlan(Math.floor(state.playhead),request.override);return;
  }
  l.busy=true;
  const begin=state.playhead;
  const source=state.motion;
  const base=Math.min(source.frames-2,Math.floor(begin));
  const frames=Math.min(64,source.frames-base);
  const motion={frames,joints:34,roots:source.roots.slice(base*3,(base+frames)*3),
    rotations:source.rotations.slice(base*136,(base+frames)*136)};
  const steps=state.kimodo ? Math.max(1,Math.min(5,Math.ceil((source.frames-1-begin)/.6))) : 5;
  try {
    const response=await api('/api/live-physics',{session:l.owner,motion,frame:begin-base,steps});
    if(!l.rig) {
      const joints=response.parents.map((parent,i)=>({parent,name:`physical_${i}`,position:response.frames[0].actual.slice(i*3,i*3+3)}));
      l.rig=new SkeletonRig(joints,{color:0x55efc4,jointColor:0xd9fff3,rootColor:0xffd166,
        emissive:0x0c5b49,rootEmissive:0x6a3b00,emissiveIntensity:.8,radius:.022,jointRadius:.034,
        rootRadius:.062,opacity:1,diamonds:false,renderOrder:7,label:'PHYSICAL',labelColor:'#55efc4'});
      l.referenceLabel=labelSprite('REFERENCE','#65a9ff',.9);scene.add(l.referenceLabel);
    }
    l.rig.setVisible(true);
    l.queue=response.frames.map((frame,i)=>({...frame,playhead:Math.min(source.frames-1,begin+(i+1)*.6)}));
    if(!l.current) {l.current=l.queue[0];l.previous=l.current;}
    l.elapsed=0;
  } catch(error) {
    l.fallen=true;document.querySelector('#live-physics-status').textContent=`Physics paused: ${error.message}. Reset explicitly to retry.`;
  } finally {l.busy=false;}
}

function advanceLivePhysics(delta) {
  const l=state.live;
  if(l.queue.length) {
    l.elapsed+=delta;
    while(l.elapsed>=.02 && l.queue.length) {
      l.elapsed-=.02;l.previous=l.current;l.current=l.queue.shift();state.playhead=l.current.playhead;
      document.documentElement.dataset.livePhysicsTime=String(l.current.time);
      if(l.current.fallen) {
        document.querySelector('#live-physics-status').textContent='Fall detected; physics continues.';
      }
    }
  }
  // Wait for computation/network without skipping physical ticks or advancing
  // the reference alone. Each new plan/clip retains the same physical session.
  if(!l.queue.length && !l.busy && l.disableRequested) {
    l.disableRequested=false;const toggle=document.querySelector('#live-physics');toggle.checked=false;toggle.dispatchEvent(new Event('change'));
  } else if(!l.queue.length && !l.busy && l.resetRequested) {
    l.resetRequested=false;document.querySelector('#live-physics-reset').click();
  } else if(!l.queue.length && !state.qaPaused) void requestLiveBatch();
}

function renderLivePhysics() {
  const l=state.live;if(!l.rig || !l.current)return;
  const next=l.queue[0]??l.current,alpha=l.queue.length?Math.min(1,l.elapsed/.02):1;
  const pose=new Float32Array(90);
  for(let i=0;i<90;i++)pose[i]=l.current.actual[i]+alpha*(next.actual[i]-l.current.actual[i]);
  l.rig.poseWorld(pose);
  l.referenceLabel.position.copy(state.rig.jointMeshes[0].position).add(new THREE.Vector3(-.25,.5,0));
  const drift=Math.hypot(l.current.actual[0]-l.current.reference[0],l.current.actual[2]-l.current.reference[2]);
  targetInfo.textContent=`Physical ${l.current.time.toFixed(2)} s · drift ${drift.toFixed(2)} m · ${l.current.contacts} contacts`;
  updateCamera();
}

async function start() {
  try {
    if (await installPhysics()) return;
    state.meta = await api('/api/meta');
    if (state.meta.runtime === 'comparison') {
      await startComparison();
      return;
    }
    if (state.meta.runtime === 'replay') {
      await startReplay();
      return;
    }
    installStyles(state.meta.styles);
    installKimodo(state.meta.kimodo_clips);
    makeSkeletons(state.meta.joints);
    installMotionQAHook();
    if(await installStream()) {requestAnimationFrame(animate);document.documentElement.dataset.testStatus='ready';return;}
    const initial = await api('/api/session', {style: state.style});
    useMotion(initial);
    await installLivePhysics();
    renderMotion(0);
    requestAnimationFrame(animate);
    if (query.get('test') === '1') await selfTest();
    else document.documentElement.dataset.testStatus = 'ready';
  } catch (error) {
    console.error(error);
    statusElement.textContent = 'Error';
    testResult.textContent = error.message;
    document.documentElement.dataset.testStatus = 'failed';
  }
}

await start();
