import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFileSync} from "node:fs";
import test from "node:test";
import {observe, predictQ, validateModel} from "../extension/policy.js";

const model = JSON.parse(readFileSync(new URL("../extension/model.json", import.meta.url)));

function gameState() {
  return {speed: 6, width: 600, trex: {x: 14, y: 93, groundY: 93, vy: 0,
    jumping: false, ducking: false, speedDrop: false, reachedMinHeight: true, width: 44},
  obstacles: []};
}

test("bundled model has the supported trained DQN contract", () => {
  validateModel(model);
  const pointer = JSON.parse(readFileSync(new URL("../models/best.json", import.meta.url)));
  const metadata = JSON.parse(readFileSync(new URL(`../models/${pointer.checkpoint}/metadata.json`, import.meta.url)));
  const zip = readFileSync(new URL(`../models/${pointer.checkpoint}/model.zip`, import.meta.url));
  assert.equal(model.checkpoint_step, pointer.steps);
  assert.equal(metadata.steps, pointer.steps);
  assert.equal(model.source_sha256, createHash("sha256").update(zip).digest("hex"));
  assert.equal(metadata.sha256, model.source_sha256);
});

test("observation excludes passed and offscreen obstacles and preserves bird features", () => {
  const state = gameState();
  const cactus = x => ({x, y: 105, width: 17, height: 35, type: "cactusSmall", speedOffset: 0});
  state.obstacles = [cactus(650), cactus(-40), {...cactus(400), type: "pterodactyl", y: 75, speedOffset: 0.8}, cactus(100)];
  const input = observe(state, 2, model.normalization);
  assert.equal(input.length, 28);
  assert.deepEqual(Array.from(input.slice(7, 10)), [0, 0, 1]);
  assert.deepEqual(Array.from(input.slice(16, 19)), [1, 0, 0]);
  assert.deepEqual(Array.from(input.slice(25, 28)), [0, 0, 1]);
  assert.equal(input[21], 0.5);
  assert.ok(Math.abs(input[24] - 6.8 / 14) < 1e-6);
});

test("Q calculation applies row-major weights, biases and ReLU", () => {
  const policy = {layers: [
    {weights: [[1, -1], [2, 1]], bias: [1, -1], activation: "relu"},
    {weights: [[2, -1]], bias: [0.5], activation: "linear"},
  ]};
  assert.deepEqual(predictQ(policy, Float32Array.from([2, 3])), [-5.5]);
});

test("corrupt models and incompatible observations fail explicitly", () => {
  assert.throws(() => validateModel(null), /Unsupported/);
  const corrupt = structuredClone(model);
  corrupt.layers[0].weights[0][0] = NaN;
  assert.throws(() => validateModel(corrupt), /invalid/);
  const changed = structuredClone(model);
  changed.schema = "unknown";
  assert.throws(() => validateModel(changed), /Unsupported/);
  const state = gameState();
  state.speed = Infinity;
  assert.throws(() => observe(state, 0, model.normalization), /incompatible/);
});
