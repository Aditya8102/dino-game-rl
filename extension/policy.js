// Keep this observation contract in parity with dino_rl/env.py.
export function observe(raw, previousAction, scales) {
  const t = raw.trex;
  const values = [raw.speed / scales.speed, (t.groundY - t.y) / scales.height, t.vy / scales.velocity,
    Number(t.jumping), Number(t.ducking), Number(t.speedDrop),
    Number(t.jumping && t.reachedMinHeight),
    ...[0, 1, 2].map(action => Number(previousAction === action))];
  const obstacles = raw.obstacles
    .filter(o => o.x < raw.width && o.x + o.width >= t.x)
    .sort((a, b) => a.x - b.x);
  const types = ["cactusSmall", "cactusLarge", "pterodactyl"];
  for (let index = 0; index < 2; index++) {
    const o = obstacles[index];
    if (!o) {
      values.push(...Array(9).fill(0));
      continue;
    }
    if (!types.includes(o.type)) throw new Error(`Unsupported Dino obstacle: ${o.type}`);
    values.push(1, (o.x - t.x - t.width) / scales.distance, o.y / scales.height, o.width / scales.distance,
      o.height / scales.height, (raw.speed + o.speedOffset) / scales.speed,
      ...types.map(type => Number(o.type === type)));
  }
  if (values.length !== 28 || !values.every(Number.isFinite)) {
    throw new Error("Chrome returned an incompatible Dino observation.");
  }
  return Float32Array.from(values);
}

export function predictQ(model, observation) {
  let values = observation;
  for (const layer of model.layers) {
    const next = new Float32Array(layer.bias.length);
    for (let row = 0; row < next.length; row++) {
      let sum = layer.bias[row];
      for (let col = 0; col < values.length; col++) sum += layer.weights[row][col] * values[col];
      next[row] = layer.activation === "relu" ? Math.max(0, sum) : sum;
    }
    values = next;
  }
  return Array.from(values);
}

export function validateModel(model) {
  if (!model || model.format_version !== 1 || model.algorithm !== "DQN" ||
      model.schema !== "dino-state-28-v1" || model.step_ms !== 50 || model.input_size !== 28 ||
      JSON.stringify(model.actions) !== '["release","jump","duck"]' ||
      !Array.isArray(model.layers) || model.layers.length !== 3 ||
      !model.game || !model.normalization ||
      !["speed", "height", "velocity", "distance"].every(key =>
        Number.isFinite(model.normalization[key]) && model.normalization[key] > 0)) {
    throw new Error("Unsupported Dino RL model contract.");
  }
  const widths = [28, 64, 64, 3];
  model.layers.forEach((layer, index) => {
    if (!layer || !Array.isArray(layer.bias) || !Array.isArray(layer.weights) ||
        layer.bias.length !== widths[index + 1] || layer.weights.length !== layer.bias.length ||
        !layer.bias.every(Number.isFinite) ||
        !layer.weights.every(row => Array.isArray(row) && row.length === widths[index] && row.every(Number.isFinite)) ||
        layer.activation !== (index === 2 ? "linear" : "relu")) {
      throw new Error("Dino RL model weights are invalid.");
    }
  });
}
