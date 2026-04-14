import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  classifyExpectedMemoryClass,
  readProcessRegistry,
  registerManagedProcess,
  unregisterManagedProcess,
} from "./process_registry.js";

test("classifyExpectedMemoryClass treats feature sequence families as high memory and ridge as low memory", () => {
  assert.equal(classifyExpectedMemoryClass("feature_lstm"), "high");
  assert.equal(classifyExpectedMemoryClass("feature_gru"), "high");
  assert.equal(classifyExpectedMemoryClass("feature_tcn"), "high");
  assert.equal(classifyExpectedMemoryClass("raw_lstm"), "high");
  assert.equal(classifyExpectedMemoryClass("ridge"), "low");
  assert.equal(classifyExpectedMemoryClass("tree_xgboost"), "low");
  assert.equal(classifyExpectedMemoryClass(undefined), "low");
});

test("registerManagedProcess writes entries and unregisterManagedProcess removes them", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-process-registry-"));
  const registryPath = path.join(tempRoot, "process_registry.json");

  try {
    await registerManagedProcess(registryPath, {
      pid: 12345,
      campaignId: "overnight-r01",
      trackId: "canonical_mainline_feature_lstm",
      taskKind: "formal_train",
      modelFamily: "feature_lstm",
      priority: 1,
      expectedMemoryClass: "high",
      command: "python scripts/train_feature_lstm.py",
      startedAt: "2026-04-08T08:00:00Z",
      cwd: "/repo",
    });

    await registerManagedProcess(registryPath, {
      pid: 12346,
      campaignId: "overnight-r01",
      trackId: "canonical_mainline_ridge",
      taskKind: "smoke_train",
      modelFamily: "ridge",
      priority: 2,
      expectedMemoryClass: "low",
      command: "python scripts/train_ridge.py",
      startedAt: "2026-04-08T08:02:00Z",
      cwd: "/repo",
    });

    const registry = await readProcessRegistry(registryPath);
    assert.equal(registry.processes.length, 2);
    assert.equal(registry.processes[0]?.pid, 12345);
    assert.equal(registry.processes[1]?.expectedMemoryClass, "low");

    await unregisterManagedProcess(registryPath, 12345);
    const after = JSON.parse(await readFile(registryPath, "utf8")) as { processes: Array<{ pid: number }> };
    assert.deepEqual(after.processes.map((item) => item.pid), [12346]);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});
