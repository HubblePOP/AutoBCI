import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { compileAutoresearchQueue } from "./queue_compiler.js";

function baseManifest() {
  return {
    review_cadence: "daily",
    tracks: [
      {
        track_id: "canonical_mainline_feature_lstm",
        topic_id: "canonical_mainline",
        runner_family: "feature_lstm",
        expected_memory_class: "high",
        internet_research_enabled: true,
        track_goal: "Protect and improve the canonical joints mainline under the fixed promotion gate with feature-sequence LSTM decoders.",
        promotion_target: "canonical_mainline",
        smoke_command: ".venv/bin/python scripts/train_feature_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 4 --batch-size 64 --seed 0 --final-eval --hidden-size 64 --num-layers 1 --patience 2 --feature-bin-ms 100.0 --feature-family lmp+hg_power --feature-reducers mean --signal-preprocess car_notch_bandpass --target-axes xyz",
        formal_command: ".venv/bin/python scripts/train_feature_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 12 --batch-size 64 --seed 0 --final-eval --hidden-size 64 --num-layers 1 --patience 3 --feature-bin-ms 100.0 --feature-family lmp+hg_power --feature-reducers mean --signal-preprocess car_notch_bandpass --target-axes xyz",
        allowed_change_scope: ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
      },
      {
        track_id: "canonical_mainline_tree_xgboost",
        topic_id: "canonical_mainline",
        runner_family: "tree_xgboost",
        expected_memory_class: "low",
        internet_research_enabled: true,
        track_goal: "Protect and cheaply sanity-check the canonical joints mainline with an XGBoost control branch.",
        promotion_target: "canonical_mainline",
        smoke_command: ".venv/bin/python scripts/train_tree_baseline.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --feature-bin-ms 100.0 --feature-family lmp+hg_power --feature-reducers mean --signal-preprocess car_notch_bandpass --model-family xgboost --xgb-n-estimators 128 --target-axes xyz",
        formal_command: ".venv/bin/python scripts/train_tree_baseline.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --feature-bin-ms 100.0 --feature-family lmp+hg_power --feature-reducers mean --signal-preprocess car_notch_bandpass --model-family xgboost --xgb-n-estimators 256 --target-axes xyz",
        allowed_change_scope: ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
      },
      {
        track_id: "relative_origin_xyz_feature_lstm",
        topic_id: "relative_origin_xyz",
        runner_family: "feature_lstm",
        expected_memory_class: "high",
        internet_research_enabled: true,
        track_goal: "Explore whether RSCA-relative xyz marker coordinates expose a better abstraction than direct joints targets with feature-sequence LSTM decoders.",
        promotion_target: "canonical_mainline",
        smoke_command: ".venv/bin/python scripts/train_feature_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_rsca_relative_xyz.yaml --epochs 4 --batch-size 64 --seed 0 --final-eval --hidden-size 64 --num-layers 1 --patience 2 --feature-bin-ms 100.0 --feature-family lmp+hg_power --feature-reducers mean --signal-preprocess car_notch_bandpass --target-axes xyz --relative-origin-marker RSCA",
        formal_command: ".venv/bin/python scripts/train_feature_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_rsca_relative_xyz.yaml --epochs 12 --batch-size 64 --seed 0 --final-eval --hidden-size 64 --num-layers 1 --patience 3 --feature-bin-ms 100.0 --feature-family lmp+hg_power --feature-reducers mean --signal-preprocess car_notch_bandpass --target-axes xyz --relative-origin-marker RSCA",
        allowed_change_scope: ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
      },
    ],
  };
}

async function writeBaseRepo(tempRoot: string): Promise<void> {
  await mkdir(path.join(tempRoot, "docs"), { recursive: true });
  await mkdir(path.join(tempRoot, "memory"), { recursive: true });
  await mkdir(path.join(tempRoot, "tools", "autoresearch"), { recursive: true });

  await writeFile(path.join(tempRoot, "docs", "CONSTITUTION.md"), "# constitution\n", "utf8");
  await writeFile(
    path.join(tempRoot, "memory", "current_strategy.md"),
    [
      "# 当前策略",
      "",
      "## 当前默认三轨",
      "",
      "1. `canonical_mainline_feature_lstm`",
      "2. `relative_origin_xyz_feature_lstm`",
      "3. `canonical_mainline_tree_xgboost`",
      "",
    ].join("\n"),
    "utf8",
  );
  await writeFile(
    path.join(tempRoot, "memory", "hermes_research_tree.md"),
    [
      "# Hermes Research Tree",
      "",
      "#### 当前主控排期",
      "",
      "任务 3：`kinematics-only / eeg-only / hybrid` 三线对照",
      "任务 4：phase-aware 便宜版本",
      "任务 8：`DMD / sDM + ridge/xgboost`",
      "",
    ].join("\n"),
    "utf8",
  );
  await writeFile(
    path.join(tempRoot, "tools", "autoresearch", "tracks.current.json"),
    `${JSON.stringify(baseManifest(), null, 2)}\n`,
    "utf8",
  );
}

test("compileAutoresearchQueue appends validated wave 1 tracks and writes runtime status", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-queue-"));
  await writeBaseRepo(tempRoot);

  const commands: string[][] = [];
  const result = await compileAutoresearchQueue({
    repoRoot: tempRoot,
    runCommand: async (command) => {
      commands.push(command);
      return {
        command,
        exitCode: 0,
        stdout: "preflight_ok\n",
        stderr: "",
      };
    },
  });

  try {
    assert.equal(result.status, "applied");
    assert.deepEqual(result.failedTrackIds, []);
    assert.deepEqual(
      result.trackIds,
      [
        "canonical_mainline_feature_lstm",
        "canonical_mainline_tree_xgboost",
        "relative_origin_xyz_feature_lstm",
        "canonical_mainline_kinematics_only_ridge",
        "canonical_mainline_hybrid_ridge",
        "canonical_mainline_phase_state_ridge",
        "canonical_mainline_dmd_sdm_ridge",
      ],
    );
    assert.equal(commands.length, 4);
    assert.ok(commands.every((command) => command.includes("--preflight-only")));

    const manifest = JSON.parse(await readFile(path.join(tempRoot, "tools", "autoresearch", "tracks.current.json"), "utf8")) as {
      tracks: Array<{ track_id: string }>;
    };
    assert.deepEqual(manifest.tracks.map((track) => track.track_id), result.trackIds);

    const runtimeState = JSON.parse(await readFile(path.join(tempRoot, "artifacts", "monitor", "autobci_remote_runtime.json"), "utf8")) as {
      queue_compiler_status: string;
      last_queue_compiler_summary: string;
      last_queue_compiler_failed_track_ids: string[];
    };
    assert.equal(runtimeState.queue_compiler_status, "applied");
    assert.match(runtimeState.last_queue_compiler_summary, /Wave 1/);
    assert.deepEqual(runtimeState.last_queue_compiler_failed_track_ids, []);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("compileAutoresearchQueue records failed wave 1 validations without dropping the base queue", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-queue-failure-"));
  await writeBaseRepo(tempRoot);

  const result = await compileAutoresearchQueue({
    repoRoot: tempRoot,
    runCommand: async (command) => {
      const joined = command.join(" ");
      return {
        command,
        exitCode: joined.includes("dmd_sdm") ? 1 : 0,
        stdout: joined.includes("dmd_sdm") ? "" : "preflight_ok\n",
        stderr: joined.includes("dmd_sdm") ? "phase recipe rejected" : "",
      };
    },
  });

  try {
    assert.equal(result.status, "applied");
    assert.deepEqual(result.failedTrackIds, ["canonical_mainline_dmd_sdm_ridge"]);
    assert.deepEqual(
      result.trackIds,
      [
        "canonical_mainline_feature_lstm",
        "canonical_mainline_tree_xgboost",
        "relative_origin_xyz_feature_lstm",
        "canonical_mainline_kinematics_only_ridge",
        "canonical_mainline_hybrid_ridge",
        "canonical_mainline_phase_state_ridge",
      ],
    );

    const runtimeState = JSON.parse(await readFile(path.join(tempRoot, "artifacts", "monitor", "autobci_remote_runtime.json"), "utf8")) as {
      last_queue_compiler_failed_track_ids: string[];
      last_queue_compiler_reason: string;
    };
    assert.deepEqual(runtimeState.last_queue_compiler_failed_track_ids, ["canonical_mainline_dmd_sdm_ridge"]);
    assert.match(runtimeState.last_queue_compiler_reason, /dmd_sdm/);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("compileAutoresearchQueue keeps an already-compiled Python Wave 1 queue without false failure", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-queue-python-wave1-"));
  await writeBaseRepo(tempRoot);

  const pythonWave1Manifest = {
    review_cadence: "daily",
    tracks: [
      { ...baseManifest().tracks[1], track_id: "canonical_mainline_tree_xgboost" },
      {
        ...baseManifest().tracks[1],
        track_id: "kinematics_only_baseline",
        topic_id: "wave1_controls",
      },
      {
        ...baseManifest().tracks[1],
        track_id: "hybrid_brain_plus_kinematics",
        topic_id: "wave1_controls",
      },
      {
        ...baseManifest().tracks[1],
        track_id: "dmd_sdm_ridge",
        topic_id: "wave1_representation",
      },
      {
        ...baseManifest().tracks[1],
        track_id: "dmd_sdm_xgboost",
        topic_id: "wave1_representation",
      },
      {
        ...baseManifest().tracks[1],
        track_id: "phase_aware_xgboost",
        topic_id: "wave1_phase_state",
      },
      {
        ...baseManifest().tracks[0],
        track_id: "phase_conditioned_feature_lstm",
        topic_id: "wave1_phase_state",
      },
      {
        ...baseManifest().tracks[1],
        track_id: "tree_calibration_catboost_or_extratrees",
        topic_id: "wave1_tree_calibration",
      },
    ],
  };
  await writeFile(
    path.join(tempRoot, "tools", "autoresearch", "tracks.current.json"),
    `${JSON.stringify(pythonWave1Manifest, null, 2)}\n`,
    "utf8",
  );

  const result = await compileAutoresearchQueue({
    repoRoot: tempRoot,
    runCommand: async (command) => ({
      command,
      exitCode: 0,
      stdout: "should_not_run\n",
      stderr: "",
    }),
  });

  try {
    assert.equal(result.status, "applied");
    assert.equal(result.failedTrackIds.length, 0);
    assert.equal(result.trackIds.length, 8);
    assert.ok(result.reason.includes("Python-side Hermes Wave 1 queue"));
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});
