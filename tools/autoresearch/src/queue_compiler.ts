import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

type Primitive = string | number | boolean | null;
type JsonValue = Primitive | JsonValue[] | { [key: string]: JsonValue };

export interface QueueCompilerOptions {
  repoRoot?: string;
  trackManifestPath?: string;
  currentStrategyPath?: string;
  researchTreePath?: string;
  runtimeStatePath?: string;
  runCommand?: (command: string[], cwd: string) => Promise<CommandResult>;
}

export interface CommandResult {
  command: string[];
  exitCode: number;
  stdout: string;
  stderr: string;
}

export interface QueueCompilerStatus {
  queue_compiler_status: "idle" | "compiling" | "validating" | "applied" | "failed";
  last_queue_compiler_summary: string;
  last_queue_compiler_reason: string;
  last_queue_compiler_track_ids: string[];
  last_queue_compiler_failed_track_ids: string[];
  last_queue_compiler_at: string;
}

export interface QueueCompilerResult {
  manifestPath: string;
  runtimeStatePath: string;
  trackIds: string[];
  failedTrackIds: string[];
  summary: string;
  reason: string;
  status: QueueCompilerStatus["queue_compiler_status"];
}

interface TrackManifest {
  review_cadence?: string;
  tracks: QueueTrack[];
}

interface QueueTrack {
  trackId: string;
  topicId?: string;
  runnerFamily?: string;
  expectedMemoryClass?: "high" | "low" | "service";
  internetResearchEnabled?: boolean;
  trackGoal: string;
  promotionTarget: string;
  smokeCommand: string;
  formalCommand: string;
  allowedChangeScope: string[];
}

interface QueueTrackRecipe {
  track: QueueTrack;
  directive: string;
  validationLabel: string;
}

const DEFAULT_TRACK_MANIFEST_PATH = "tools/autoresearch/tracks.current.json";
const DEFAULT_CURRENT_STRATEGY_PATH = "memory/current_strategy.md";
const DEFAULT_RESEARCH_TREE_PATH = "memory/hermes_research_tree.md";
const DEFAULT_RUNTIME_STATE_PATH = "artifacts/monitor/autobci_remote_runtime.json";

const BASE_TRACK_IDS = [
  "canonical_mainline_feature_lstm",
  "relative_origin_xyz_feature_lstm",
  "canonical_mainline_tree_xgboost",
];

const PYTHON_WAVE1_TRACK_IDS = [
  "phase_conditioned_feature_lstm",
  "phase_aware_xgboost",
  "dmd_sdm_ridge",
  "dmd_sdm_xgboost",
  "canonical_mainline_tree_xgboost",
  "hybrid_brain_plus_kinematics",
  "kinematics_only_baseline",
  "tree_calibration_catboost_or_extratrees",
];

const WAVE1_DIRECTIVE_PATTERNS = [
  "kinematics-only / eeg-only / hybrid",
  "phase-aware 便宜版本",
  "DMD / sDM + ridge/xgboost",
];

const WAVE1_RECIPES: QueueTrackRecipe[] = [
  {
    directive: WAVE1_DIRECTIVE_PATTERNS[0] ?? "",
    validationLabel: "kinematics_only_ridge",
    track: {
      trackId: "canonical_mainline_kinematics_only_ridge",
      topicId: "canonical_mainline",
      runnerFamily: "ridge",
      expectedMemoryClass: "low",
      internetResearchEnabled: true,
      trackGoal: "Wave 1 task 3: compare the kinematics-only control line against the EEG-bearing mainline.",
      promotionTarget: "canonical_mainline",
      smokeCommand:
        ".venv/bin/python scripts/train_control_baseline.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --control-mode kinematics_only --model-family ridge --ridge-alpha 1.0 --feature-bin-ms 100.0 --feature-reducers mean --signal-preprocess car_notch_bandpass --feature-family lmp+hg_power --target-axes xyz",
      formalCommand:
        ".venv/bin/python scripts/train_control_baseline.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --control-mode kinematics_only --model-family ridge --ridge-alpha 1.0 --feature-bin-ms 100.0 --feature-reducers mean --signal-preprocess car_notch_bandpass --feature-family lmp+hg_power --target-axes xyz",
      allowedChangeScope: ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
    },
  },
  {
    directive: WAVE1_DIRECTIVE_PATTERNS[0] ?? "",
    validationLabel: "hybrid_ridge",
    track: {
      trackId: "canonical_mainline_hybrid_ridge",
      topicId: "canonical_mainline",
      runnerFamily: "ridge",
      expectedMemoryClass: "low",
      internetResearchEnabled: true,
      trackGoal: "Wave 1 task 3: compare a hybrid EEG + kinematics control line against the kinematics-only line.",
      promotionTarget: "canonical_mainline",
      smokeCommand:
        ".venv/bin/python scripts/train_control_baseline.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --control-mode hybrid --model-family ridge --ridge-alpha 1.0 --feature-bin-ms 100.0 --feature-reducers mean --signal-preprocess car_notch_bandpass --feature-family lmp+hg_power --target-axes xyz",
      formalCommand:
        ".venv/bin/python scripts/train_control_baseline.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --control-mode hybrid --model-family ridge --ridge-alpha 1.0 --feature-bin-ms 100.0 --feature-reducers mean --signal-preprocess car_notch_bandpass --feature-family lmp+hg_power --target-axes xyz",
      allowedChangeScope: ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
    },
  },
  {
    directive: WAVE1_DIRECTIVE_PATTERNS[1] ?? "",
    validationLabel: "phase_state_ridge",
    track: {
      trackId: "canonical_mainline_phase_state_ridge",
      topicId: "canonical_mainline",
      runnerFamily: "ridge",
      expectedMemoryClass: "low",
      internetResearchEnabled: true,
      trackGoal: "Wave 1 task 4: test whether phase-state features improve the canonical mainline with a cheap ridge baseline.",
      promotionTarget: "canonical_mainline",
      smokeCommand:
        ".venv/bin/python scripts/train_ridge.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --feature-bin-ms 100.0 --feature-family phase_state --feature-reducers mean --signal-preprocess car_notch_bandpass --ridge-alpha 1.0 --target-axes xyz",
      formalCommand:
        ".venv/bin/python scripts/train_ridge.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --feature-bin-ms 100.0 --feature-family phase_state --feature-reducers mean --signal-preprocess car_notch_bandpass --ridge-alpha 1.0 --target-axes xyz",
      allowedChangeScope: ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
    },
  },
  {
    directive: WAVE1_DIRECTIVE_PATTERNS[2] ?? "",
    validationLabel: "dmd_sdm_ridge",
    track: {
      trackId: "canonical_mainline_dmd_sdm_ridge",
      topicId: "canonical_mainline",
      runnerFamily: "ridge",
      expectedMemoryClass: "low",
      internetResearchEnabled: true,
      trackGoal: "Wave 1 task 8: test whether DMD/sDM-style features are better than the current simple feature baseline.",
      promotionTarget: "canonical_mainline",
      smokeCommand:
        ".venv/bin/python scripts/train_ridge.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --feature-bin-ms 100.0 --feature-family dmd_sdm --feature-reducers mean --signal-preprocess car_notch_bandpass --ridge-alpha 1.0 --target-axes xyz",
      formalCommand:
        ".venv/bin/python scripts/train_ridge.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 1 --batch-size 0 --seed 0 --final-eval --feature-bin-ms 100.0 --feature-family dmd_sdm --feature-reducers mean --signal-preprocess car_notch_bandpass --ridge-alpha 1.0 --target-axes xyz",
      allowedChangeScope: ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
    },
  },
];

export async function compileAutoresearchQueue(options: QueueCompilerOptions = {}): Promise<QueueCompilerResult> {
  const repoRoot = options.repoRoot ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
  const manifestPath = resolveRepoPath(repoRoot, options.trackManifestPath ?? DEFAULT_TRACK_MANIFEST_PATH);
  const currentStrategyPath = resolveRepoPath(repoRoot, options.currentStrategyPath ?? DEFAULT_CURRENT_STRATEGY_PATH);
  const researchTreePath = resolveRepoPath(repoRoot, options.researchTreePath ?? DEFAULT_RESEARCH_TREE_PATH);
  const runtimeStatePath = resolveRepoPath(repoRoot, options.runtimeStatePath ?? DEFAULT_RUNTIME_STATE_PATH);
  const runCommand = options.runCommand ?? runSpawnCommand;

  const [manifestText, currentStrategyText, researchTreeText] = await Promise.all([
    readFile(manifestPath, "utf8"),
    readFile(currentStrategyPath, "utf8"),
    readFile(researchTreePath, "utf8"),
  ]);

  const manifest = parseManifest(manifestText, manifestPath);
  const baseTrackIds = manifest.tracks.map((track) => track.trackId);
  const hasPythonWave1Queue =
    PYTHON_WAVE1_TRACK_IDS.filter((trackId) => baseTrackIds.includes(trackId)).length >= 4 &&
    baseTrackIds.includes("canonical_mainline_tree_xgboost");
  if (hasPythonWave1Queue) {
    const summary = "当前执行队列已经是 Hermes 编译出的 Wave 1 队列，保持不变。";
    const reason = "Detected a Python-side Hermes Wave 1 queue; skip legacy queue_compiler regeneration.";
    await writeQueueCompilerStatus(runtimeStatePath, {
      queue_compiler_status: "applied",
      last_queue_compiler_summary: summary,
      last_queue_compiler_reason: reason,
      last_queue_compiler_track_ids: baseTrackIds,
      last_queue_compiler_failed_track_ids: [],
      last_queue_compiler_at: new Date().toISOString(),
    });
    return {
      manifestPath,
      runtimeStatePath,
      trackIds: baseTrackIds,
      failedTrackIds: [],
      summary,
      reason,
      status: "applied",
    };
  }
  const baseTrackMismatch = BASE_TRACK_IDS.filter((trackId) => !baseTrackIds.includes(trackId));
  const strategyTrackMismatch = BASE_TRACK_IDS.filter((trackId) => !currentStrategyText.includes(trackId));
  const mismatchTrackIds = dedupeStrings([...baseTrackMismatch, ...strategyTrackMismatch]);
  if (mismatchTrackIds.length > 0) {
    const reason = `current_strategy and tracks.current are out of sync; missing base tracks: ${mismatchTrackIds.join(", ")}`;
    await writeQueueCompilerStatus(runtimeStatePath, {
      queue_compiler_status: "failed",
      last_queue_compiler_summary: "当前默认三轨和执行 manifest 不一致，未写入新的执行队列。",
      last_queue_compiler_reason: reason,
      last_queue_compiler_track_ids: baseTrackIds,
      last_queue_compiler_failed_track_ids: mismatchTrackIds,
      last_queue_compiler_at: new Date().toISOString(),
    });
    return {
      manifestPath,
      runtimeStatePath,
      trackIds: baseTrackIds,
      failedTrackIds: mismatchTrackIds,
      summary: "当前默认三轨和执行 manifest 不一致，未写入新的执行队列。",
      reason,
      status: "failed",
    };
  }

  const candidateRecipes = buildCandidateRecipes({ currentStrategyText, researchTreeText });
  const validatedAdditions: QueueTrack[] = [];
  const failedTrackIds: string[] = [];
  const failedReasons: string[] = [];

  for (const recipe of candidateRecipes) {
    const validation = await validateTrack(recipe, { repoRoot, runCommand });
    if (validation.ok) {
      validatedAdditions.push(recipe.track);
      continue;
    }
    failedTrackIds.push(recipe.track.trackId);
    failedReasons.push(`${recipe.track.trackId}: ${validation.reason}`);
  }

  const mergedTracks = dedupeTracks([...manifest.tracks, ...validatedAdditions]);
  const outputManifest = {
    review_cadence: manifest.review_cadence ?? "daily",
    tracks: mergedTracks.map(serializeTrack),
  };

  await writeFile(manifestPath, `${JSON.stringify(outputManifest, null, 2)}\n`, "utf8");

  const summary =
    validatedAdditions.length > 0
      ? `从 Research Tree 编译出 ${validatedAdditions.length} 条可执行 Wave 1 方向，已写入 tracks.current.json。`
      : "Research Tree 没有发现可执行的 Wave 1 方向，保持原有执行队列。";
  const reason =
    failedReasons.length > 0
      ? `${summary} 验证失败：${failedReasons.join("；")}`
      : `${summary} 所有候选都通过了预检。`;
  const status: QueueCompilerStatus["queue_compiler_status"] = "applied";

  await writeQueueCompilerStatus(runtimeStatePath, {
    queue_compiler_status: status,
    last_queue_compiler_summary: summary,
    last_queue_compiler_reason: reason,
    last_queue_compiler_track_ids: mergedTracks.map((track) => track.trackId),
    last_queue_compiler_failed_track_ids: failedTrackIds,
    last_queue_compiler_at: new Date().toISOString(),
  });

  return {
    manifestPath,
    runtimeStatePath,
    trackIds: mergedTracks.map((track) => track.trackId),
    failedTrackIds,
    summary,
    reason,
    status,
  };
}

export function shouldCompileAutoresearchQueue(argv: string[]): boolean {
  const manifestPath = getArgValue(argv, "--track-manifest") ?? getArgValue(argv, "--track_manifest");
  if (!manifestPath) {
    return true;
  }
  const normalized = manifestPath.replaceAll("\\", "/");
  return normalized.endsWith(DEFAULT_TRACK_MANIFEST_PATH);
}

async function validateTrack(
  recipe: QueueTrackRecipe,
  options: { repoRoot: string; runCommand: (command: string[], cwd: string) => Promise<CommandResult> },
): Promise<{ trackId: string; ok: boolean; reason: string }> {
  const tempDir = path.join(options.repoRoot, "artifacts", "monitor", "queue_compiler_preflight");
  await mkdir(tempDir, { recursive: true });
  const outputPath = path.join(tempDir, `${recipe.validationLabel}.json`);
  const command = recipe.track.smokeCommand.split(/\s+/).filter(Boolean);
  command.push("--preflight-only", "--output-json", outputPath);
  const result = await options.runCommand(command, options.repoRoot);
  if (result.exitCode === 0) {
    return {
      trackId: recipe.track.trackId,
      ok: true,
      reason: "preflight_ok",
    };
  }
  return {
    trackId: recipe.track.trackId,
    ok: false,
    reason: result.stderr.trim() || result.stdout.trim() || "preflight failed",
  };
}

async function runSpawnCommand(command: string[], cwd: string): Promise<CommandResult> {
  const [executable, ...args] = command;
  if (!executable) {
    throw new Error("Cannot run an empty command.");
  }
  return await new Promise<CommandResult>((resolve, reject) => {
    const child = spawn(executable, args, {
      cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
      shell: false,
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({
        command,
        exitCode: code ?? 1,
        stdout,
        stderr,
      });
    });
  });
}

function buildCandidateRecipes(context: { currentStrategyText: string; researchTreeText: string }): QueueTrackRecipe[] {
  const text = `${context.currentStrategyText}\n${context.researchTreeText}`;
  return WAVE1_RECIPES.filter((recipe) => text.includes(recipe.directive));
}

function parseManifest(text: string, manifestPath: string): TrackManifest {
  const parsed = JSON.parse(text) as { review_cadence?: unknown; tracks?: unknown };
  if (!Array.isArray(parsed.tracks) || parsed.tracks.length === 0) {
    throw new Error(`Track manifest ${manifestPath} must contain at least one track.`);
  }
  return {
    review_cadence: typeof parsed.review_cadence === "string" && parsed.review_cadence.trim() ? parsed.review_cadence : "daily",
    tracks: parsed.tracks.map(normalizeTrackRecord),
  };
}

function normalizeTrackRecord(value: unknown): QueueTrack {
  if (!value || typeof value !== "object") {
    throw new Error("Track manifest entries must be objects.");
  }
  const record = value as Record<string, unknown>;
  const trackId = readString(record.trackId) ?? readString(record.track_id);
  const trackGoal = readString(record.trackGoal) ?? readString(record.track_goal);
  const promotionTarget = readString(record.promotionTarget) ?? readString(record.promotion_target);
  const smokeCommand = readString(record.smokeCommand) ?? readString(record.smoke_command);
  const formalCommand = readString(record.formalCommand) ?? readString(record.formal_command);
  if (!trackId || !trackGoal || !promotionTarget || !smokeCommand || !formalCommand) {
    throw new Error("Track manifest entries must define track_id, track_goal, promotion_target, smoke_command, and formal_command.");
  }
  return {
    trackId,
    topicId: readString(record.topicId) ?? readString(record.topic_id) ?? undefined,
    runnerFamily: readString(record.runnerFamily) ?? readString(record.runner_family) ?? undefined,
    expectedMemoryClass: readMemoryClass(record.expectedMemoryClass) ?? readMemoryClass(record.expected_memory_class) ?? undefined,
    internetResearchEnabled: readBoolean(record.internetResearchEnabled) ?? readBoolean(record.internet_research_enabled) ?? undefined,
    trackGoal,
    promotionTarget,
    smokeCommand,
    formalCommand,
    allowedChangeScope: readStringList(record.allowedChangeScope) ?? readStringList(record.allowed_change_scope) ?? [],
  };
}

function dedupeTracks(tracks: QueueTrack[]): QueueTrack[] {
  const seen = new Set<string>();
  const deduped: QueueTrack[] = [];
  for (const track of tracks) {
    if (seen.has(track.trackId)) {
      continue;
    }
    seen.add(track.trackId);
    deduped.push(normalizeTrackRecord(track));
  }
  return deduped;
}

function serializeTrack(track: QueueTrack): Record<string, JsonValue> {
  return {
    track_id: track.trackId,
    topic_id: track.topicId ?? null,
    runner_family: track.runnerFamily ?? null,
    expected_memory_class: track.expectedMemoryClass ?? null,
    internet_research_enabled: track.internetResearchEnabled ?? null,
    track_goal: track.trackGoal,
    promotion_target: track.promotionTarget,
    smoke_command: track.smokeCommand,
    formal_command: track.formalCommand,
    allowed_change_scope: track.allowedChangeScope,
  };
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readMemoryClass(value: unknown): QueueTrack["expectedMemoryClass"] | null {
  return value === "high" || value === "low" || value === "service" ? value : null;
}

function readStringList(value: unknown): string[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const items = value.filter((item): item is string => typeof item === "string" && item.trim() !== "");
  return items.length > 0 ? items : [];
}

async function writeQueueCompilerStatus(runtimeStatePath: string, status: QueueCompilerStatus): Promise<void> {
  await mkdir(path.dirname(runtimeStatePath), { recursive: true });
  let existing: Record<string, JsonValue> = {};
  try {
    existing = JSON.parse(await readFile(runtimeStatePath, "utf8")) as Record<string, JsonValue>;
  } catch {
    existing = {};
  }
  const merged = {
    ...existing,
    ...status,
  };
  await writeFile(runtimeStatePath, `${JSON.stringify(merged, null, 2)}\n`, "utf8");
}

function resolveRepoPath(repoRoot: string, maybePath: string): string {
  return path.isAbsolute(maybePath) ? maybePath : path.join(repoRoot, maybePath);
}

function getArgValue(argv: string[], flag: string): string | null {
  const index = argv.indexOf(flag);
  if (index === -1) {
    return null;
  }
  const candidate = argv[index + 1];
  if (!candidate || candidate.startsWith("--")) {
    return null;
  }
  return candidate;
}

function dedupeStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter((value) => value.trim() !== "")));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await compileAutoresearchQueue();
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}
