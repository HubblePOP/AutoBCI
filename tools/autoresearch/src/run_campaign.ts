import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { Codex, type FileChangeItem, type ThreadItem } from "@openai/codex-sdk";
import {
  buildCodexPrompt,
  buildDailyReviewPacket,
  decideTrackOutcome,
  loadProgramDocuments,
  loadTrackManifest,
  renderDailyReviewPacketMarkdown,
  type CampaignTrack,
} from "./campaign_program.js";
import {
  applyRuntimeCampaignOverlay,
  loadRuntimeCampaignOverlay,
  type RuntimeCampaignTrack,
  type RuntimeTrackOrigin,
} from "./runtime_campaign.js";
import { sanitizeLaunchEnvironment } from "./launch_support.js";
import { registerManagedProcess, unregisterManagedProcess } from "./process_registry.js";

type Primitive = string | number | boolean | null;
type JsonValue = Primitive | JsonValue[] | { [key: string]: JsonValue };

type Stage =
  | "baseline"
  | "bank_qc"
  | "bank_qc_failed"
  | "editing"
  | "smoke"
  | "formal_eval"
  | "accepted"
  | "rejected"
  | "rollback"
  | "done"
  | "paused";

type CampaignMode = "exploration" | "closeout";
type BudgetState = "healthy" | "nearing_cap" | "capped";
type StopReason = "none" | "no_improvement" | "wall_clock_cap" | "manual_stop_loss";

interface MetricSnapshot {
  source_path: string;
  result_json: string;
  dataset_name: string;
  target_mode: string;
  target_space: string;
  primary_metric_name: string;
  val_primary_metric: number | null;
  formal_val_primary_metric: number | null;
  val_rmse: number | null;
  test_primary_metric: number | null;
  test_rmse: number | null;
  best_checkpoint_path: string | null;
  last_checkpoint_path: string | null;
  feature_family: string | null;
  model_family: string | null;
  experiment_track: string | null;
  evaluation_mode: string | null;
  artifacts: string[];
}

interface AcceptedBestSnapshot {
  run_id: string;
  dataset_name: string;
  target_mode: string;
  target_space: string;
  primary_metric_name: string;
  val_primary_metric: number | null;
  formal_val_primary_metric: number | null;
  val_rmse: number | null;
  test_primary_metric: number | null;
  test_rmse: number | null;
  feature_family: string | null;
  model_family: string | null;
  evaluation_mode: string | null;
  result_json: string | null;
  artifacts: string[];
}

interface CandidateSnapshot {
  run_id: string;
  stage: string;
  track_id: string;
  track_goal: string;
  promotion_target: string;
  hypothesis: string;
  why_this_change: string;
  changes_summary: string;
  change_bucket: string;
  track_comparison_note: string;
  files_touched: string[];
  commands: string[];
  search_queries: ResearchQuery[];
  research_evidence: ResearchEvidence[];
  smoke_metrics: MetricSnapshot | null;
  final_metrics: MetricSnapshot | null;
  allowed_scope_ok: boolean;
  rollback_applied: boolean;
  relevance_label: string;
  relevance_reason: string;
  decision: string;
  next_step: string;
  artifacts: string[];
  tool_usage_summary: ThreadItemUsageSummary | null;
  thinking_heartbeat_at: string;
  last_retrieval_at: string;
  last_decision_at: string;
  last_judgment_at: string;
  last_materialization_at: string;
  last_smoke_at: string;
  stale_reason_codes: string[];
  pivot_reason_codes: string[];
  search_budget_state: BudgetState;
}

interface TrackRuntimeState {
  track_id: string;
  track_goal: string;
  promotion_target: string;
  smoke_command: string;
  formal_command: string;
  allowed_change_scope: string[];
  track_origin: RuntimeTrackOrigin;
  force_fresh_thread: boolean;
  current_iteration: number;
  patience_streak: number;
  edit_turns_used: number;
  formal_runs_completed: number;
  stage: Stage;
  last_run_id: string;
  last_decision: string;
  codex_thread_id: string | null;
  local_best: AcceptedBestSnapshot;
  latest_run_id: string;
  latest_smoke_run_id: string;
  latest_formal_run_id: string;
  latest_val_primary_metric: number | null;
  latest_test_primary_metric: number | null;
  latest_val_rmse: number | null;
  latest_test_rmse: number | null;
  best_val_primary_metric: number | null;
  best_test_primary_metric: number | null;
  best_val_rmse: number | null;
  best_test_rmse: number | null;
  last_result_summary: string;
  method_variant_label: string;
  input_mode_label: string;
  series_class: string;
  promotable: boolean;
  tool_usage_summary: ThreadItemUsageSummary | null;
  thinking_heartbeat_at: string;
  last_retrieval_at: string;
  last_decision_at: string;
  last_judgment_at: string;
  last_materialization_at: string;
  last_smoke_at: string;
  stale_reason_codes: string[];
  pivot_reason_codes: string[];
  search_budget_state: BudgetState;
  updated_at: string;
}

interface CampaignStatus {
  campaign_id: string;
  started_at: string;
  current_iteration: number;
  max_iterations: number;
  patience: number;
  stage: Stage;
  active_track_id: string | null;
  frozen_baseline: AcceptedBestSnapshot;
  accepted_stable_best: AcceptedBestSnapshot;
  leading_unverified_candidate: AcceptedBestSnapshot;
  accepted_best: AcceptedBestSnapshot;
  candidate: CandidateSnapshot;
  track_states: TrackRuntimeState[];
  current_command: string;
  updated_at: string;
  campaign_mode: CampaignMode;
  budget_state: BudgetState;
  stop_reason: StopReason;
  codex_thread_id?: string | null;
  patience_streak?: number;
  last_error?: string | null;
  tool_usage_summary: ThreadItemUsageSummary | null;
  thinking_heartbeat_at: string;
  last_retrieval_at: string;
  last_decision_at: string;
  last_judgment_at: string;
  last_materialization_at: string;
  last_smoke_at: string;
  stale_reason_codes: string[];
  pivot_reason_codes: string[];
  search_budget_state: BudgetState;
}

interface CampaignRecord {
  campaign_id: string;
  run_id: string;
  parent_run_id: string | null;
  iteration: number;
  stage: string;
  recorded_at: string;
  agent_name: string;
  track_id: string;
  track_goal: string;
  promotion_target: string;
  dataset_name: string;
  target_mode: string;
  target_space: string;
  primary_metric_name: string;
  hypothesis: string;
  why_this_change: string;
  changes_summary: string;
  change_bucket: string;
  track_comparison_note: string;
  files_touched: string[];
  commands: string[];
  search_queries: ResearchQuery[];
  research_evidence: ResearchEvidence[];
  smoke_metrics: MetricSnapshot | null;
  final_metrics: MetricSnapshot | null;
  allowed_scope_ok: boolean;
  rollback_applied: boolean;
  relevance_label: string;
  relevance_reason: string;
  decision: string;
  next_step: string;
  artifacts: string[];
  tool_usage_summary: ThreadItemUsageSummary | null;
  thinking_heartbeat_at: string;
  last_retrieval_at: string;
  last_decision_at: string;
  last_judgment_at: string;
  last_materialization_at: string;
  last_smoke_at: string;
  stale_reason_codes: string[];
  pivot_reason_codes: string[];
  search_budget_state: BudgetState;
}

interface ParsedArgs {
  [key: string]: string | boolean | undefined;
}

interface RunCommandResult {
  command: string;
  exitCode: number;
  stdout: string;
  stderr: string;
  durationMs: number;
}

interface SnapshotEntry {
  existed: boolean;
  content: string | null;
}

interface CodexEditResult {
  proposal: {
    hypothesis: string;
    why_this_change: string;
    changes_summary: string;
    change_bucket: string;
    track_comparison_note: string;
    files_touched: string[];
    next_step: string;
    search_queries: ResearchQuery[];
    research_evidence: ResearchEvidence[];
  };
  items: ThreadItem[];
  threadId: string | null;
}

interface ResearchQuery {
  search_query: string;
  search_intent: string;
}

interface ResearchEvidence extends ResearchQuery {
  source_type: string;
  source_url: string;
  source_title: string;
  why_it_matters: string;
}

interface ThreadItemUsageSummary {
  total_items: number;
  reasoning_items: number;
  agent_messages: number;
  command_executions: number;
  file_changes: number;
  web_searches: number;
  mcp_tool_calls: number;
  todo_lists: number;
  errors: number;
  completed_items: number;
  failed_items: number;
}

interface TurnTelemetryState {
  tool_usage_summary: ThreadItemUsageSummary | null;
  thinking_heartbeat_at: string;
  last_retrieval_at: string;
  last_decision_at: string;
  last_judgment_at: string;
  last_materialization_at: string;
  last_smoke_at: string;
  stale_reason_codes: string[];
  pivot_reason_codes: string[];
  search_budget_state: BudgetState;
}

interface ResearchQueryLogEntry extends ResearchQuery {
  campaign_id: string;
  run_id: string;
  track_id: string;
  recorded_at: string;
  used_in_run_id: string;
}

interface ResearchEvidenceLogEntry extends ResearchEvidence {
  campaign_id: string;
  run_id: string;
  track_id: string;
  recorded_at: string;
  used_in_run_id: string;
}

interface RelevanceClassification {
  label: "on_track" | "supporting_change" | "exploratory_but_indirect" | "off_track_but_ran";
  reason: string;
  hardBlock: boolean;
  relevantPatterns: string[];
  coreFiles: string[];
  supportingFiles: string[];
  offTrackFiles: string[];
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TOOLS_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(TOOLS_ROOT, "..", "..");

const STATUS_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "autoresearch_status.json");
const TOOLS_LEDGER_PATH = path.join(TOOLS_ROOT, "experiment_ledger.jsonl");
const MONITOR_LEDGER_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "experiment_ledger.jsonl");
const RESEARCH_QUERIES_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "research_queries.jsonl");
const RESEARCH_EVIDENCE_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "research_evidence.jsonl");
const PROCESS_REGISTRY_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "process_registry.json");
const DEFAULT_TRACK_MANIFEST_PATH = path.join(TOOLS_ROOT, "tracks.current.json");
const DEFAULT_REVIEW_PACKET_DIR = path.join(REPO_ROOT, "artifacts", "monitor", "review_packets");
const DEFAULT_BASELINE_METRICS_PATH = path.join(REPO_ROOT, "artifacts", "walk_matched_v1_64clean_joints_baseline_000.json");
const DEFAULT_SMOKE_OUTPUT_DIR = path.join(REPO_ROOT, "artifacts", "monitor", "autoresearch_runs");
const DEFAULT_PYTHON = path.join(REPO_ROOT, ".venv", "bin", "python");
const DEFAULT_BASELINE_CHECKPOINT_PATH = path.join(
  REPO_ROOT,
  "artifacts",
  "checkpoints",
  "walk_matched_v1_64clean_joints_baseline_000_best_val.pt",
);

const DEFAULT_BASELINE_COMMAND =
  `${DEFAULT_PYTHON} scripts/train_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 1 --final-eval --stride-samples 2000 --hidden-size 32 --num-layers 1 --batch-size 32 --output-json ${shellQuote(DEFAULT_BASELINE_METRICS_PATH)} --checkpoint-path ${shellQuote(DEFAULT_BASELINE_CHECKPOINT_PATH)}`;
const DEFAULT_SMOKE_COMMAND =
  `${DEFAULT_PYTHON} scripts/train_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --final-eval --stride-samples 2000 --hidden-size 32 --num-layers 1 --batch-size 32`;
const DEFAULT_FORMAL_COMMAND =
  `${DEFAULT_PYTHON} scripts/train_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --epochs 1 --final-eval --stride-samples 2000 --hidden-size 32 --num-layers 1 --batch-size 32`;
const DEFAULT_BANK_QC_COMMAND =
  `${DEFAULT_PYTHON} scripts/run_bank_qc_gate.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --strict`;

const DEFAULT_ALLOWED_DIRS = [
  path.join(REPO_ROOT, "scripts"),
  path.join(REPO_ROOT, "src", "bci_autoresearch", "models"),
  path.join(REPO_ROOT, "src", "bci_autoresearch", "features"),
];

const DEFAULT_AGENT_NAME = "autoresearch-campaign-codex-sdk";
const DEFAULT_PRIMARY_METRIC_NAME = "val_metrics.mean_pearson_r_zero_lag_macro";
const DEFAULT_CANONICAL_DATASET_NAME = "walk_matched_v1_64clean_joints";
const MIN_IMPROVEMENT_EPSILON = 1e-9;
const WALL_CLOCK_BUDGET_MS = 4 * 60 * 60 * 1000;
const NEARING_CAP_RATIO = 0.75;
const MAX_EDIT_TURNS_PER_TRACK = 2;
const NO_IMPROVEMENT_LIMIT = 2;
const AUTORESEARCH_EXPLORATION_REASONING_EFFORT = "medium";
const AUTORESEARCH_CLOSEOUT_REASONING_EFFORT = "low";

const ALLOWLIST = [
  /^scripts\/train_[^/]+\.py$/,
  /^scripts\/build_monitor_artifacts\.py$/,
  /^scripts\/analyze_channel_halves\.py$/,
  /^src\/bci_autoresearch\/models\/.+$/,
  /^src\/bci_autoresearch\/features\/.+$/,
];

const SUPPORTING_ALLOWLIST = [
  /^tests\/test_[^/]+\.py$/,
  /^tests\/test_[^/]+\.ts$/,
];

const DENYLIST_PATHS = [
  /^scripts\/convert_session\.py$/,
  /^src\/bci_autoresearch\/data\/.+$/,
  /^data\/.+$/,
  /^\/Volumes\/.+$/,
];

export function campaignReasoningEffortForMode(mode: CampaignMode): string {
  return mode === "closeout"
    ? AUTORESEARCH_CLOSEOUT_REASONING_EFFORT
    : AUTORESEARCH_EXPLORATION_REASONING_EFFORT;
}

export function shouldTrackSkipCodexEdit(track: CampaignTrack): boolean {
  return Boolean(track.skipCodexEdit || track.validated);
}

function buildValidatedTrackBypassResult(track: CampaignTrack): CodexEditResult {
  return {
    proposal: {
      hypothesis: "这条轨在进入 campaign 前已经过预验证，先直接看同口径 smoke/formal 是否能出分。",
      why_this_change: "moonshot 轨已经完成 preflight，不需要再经过一轮通用代码编辑，直接进入同试次纯脑电验证更快。",
      changes_summary: `跳过 codex edit，直接执行 ${track.runnerFamily ?? track.trackId} 的 smoke/formal 验证。`,
      change_bucket: "model-led",
      track_comparison_note: "This track was prevalidated outside the generic codex edit loop.",
      files_touched: [],
      next_step: "直接进入 smoke；如果 smoke 优于本地最优，再进 formal。",
      search_queries: [],
      research_evidence: [],
    },
    items: [],
    threadId: null,
  };
}

function buildValidatedTrackAudit(): {
  allowedScopeOk: boolean;
  filesTouched: string[];
  fileChanges: Array<{ path: string; kind: string }>;
  violations: string[];
} {
  return {
    allowedScopeOk: true,
    filesTouched: [],
    fileChanges: [],
    violations: [],
  };
}

function buildValidatedTrackRelevance(): RelevanceClassification {
  return {
    label: "on_track",
    reason: "这条轨在进入 campaign 前已经通过预验证，所以本轮直接把预算花在 smoke/formal，而不是重复 codex edit。",
    hardBlock: false,
    relevantPatterns: [],
    coreFiles: [],
    supportingFiles: [],
    offTrackFiles: [],
  };
}

function buildCampaignCodex(mode: CampaignMode): Codex {
  return new Codex({
    config: {
      model_reasoning_effort: campaignReasoningEffortForMode(mode),
    },
  });
}

function inferTrackSeriesClass(trackId: string): string {
  const normalized = trackId.trim().toLowerCase();
  if (
    normalized.includes("kinematics_only")
    || normalized.includes("hybrid_brain_plus_kinematics")
    || normalized.includes("tree_calibration")
  ) {
    return "control";
  }
  if (normalized.includes("relative_origin_xyz_upper_bound")) {
    return "same_session_reference";
  }
  if (normalized.includes("relative_origin_xyz") || normalized.includes("phase_aware")) {
    return "structure";
  }
  return "mainline_brain";
}

function inferTrackMethodVariantLabel(trackId: string, seriesClass: string): string {
  const normalized = trackId.trim().toLowerCase();
  if (normalized.includes("kinematics_only")) {
    return "只用运动学历史，不用脑电";
  }
  if (normalized.includes("hybrid_brain_plus_kinematics")) {
    return "混合输入（脑电 + 运动学历史）";
  }
  if (normalized.includes("tree_calibration")) {
    return "树模型校准（Extra Trees）";
  }
  if (normalized.includes("phase_conditioned")) {
    return "phase 条件版";
  }
  if (normalized.includes("phase_aware")) {
    return "phase-aware 特征";
  }
  if (normalized.includes("dmd_sdm")) {
    return "DMD/sDM 特征";
  }
  if (seriesClass === "same_session_reference") {
    return "相对 RSCA 同试次参考";
  }
  if (seriesClass === "structure") {
    return "相对 RSCA 三方向坐标";
  }
  if (normalized.startsWith("canonical_mainline")) {
    return "标准主线";
  }
  return trackId;
}

function inferTrackInputModeLabel(trackId: string, seriesClass: string): string {
  const normalized = trackId.trim().toLowerCase();
  if (normalized.includes("kinematics_only")) {
    return "只用运动学历史，不用脑电";
  }
  if (normalized.includes("hybrid_brain_plus_kinematics") || normalized.includes("tree_calibration")) {
    return "脑电 + 运动学历史";
  }
  if (seriesClass === "same_session_reference") {
    return "只用脑电（同试次参考）";
  }
  return "只用脑电";
}

function isPromotableSeriesClass(seriesClass: string): boolean {
  return seriesClass === "mainline_brain";
}

function humanizeTrackModelFamily(modelFamily: string | null | undefined): string {
  const normalized = String(modelFamily || "").trim().toLowerCase();
  if (!normalized) {
    return "未标注算法";
  }
  if (normalized.includes("feature_gru")) {
    return "Feature GRU";
  }
  if (normalized.includes("feature_tcn")) {
    return "Feature TCN";
  }
  if (normalized.includes("feature_lstm") || normalized.includes("lstm")) {
    return "Feature LSTM";
  }
  if (normalized.includes("extra") && normalized.includes("tree")) {
    return "Extra Trees";
  }
  if (normalized.includes("catboost")) {
    return "CatBoost";
  }
  if (normalized.includes("xgboost")) {
    return "XGBoost";
  }
  if (normalized.includes("ridge")) {
    return "Ridge";
  }
  return modelFamily || "未标注算法";
}

function humanizeTrackDecision(decision: string, promotable: boolean): string {
  const normalized = decision.trim().toLowerCase();
  if (normalized === "hold_for_promotion_review") {
    return "进入候选复审";
  }
  if (normalized === "hold_for_packet_gate") {
    return promotable ? "已正式比较" : "控制实验，不进入主线晋升";
  }
  if (normalized === "accepted") {
    return "已正式比较";
  }
  if (normalized === "rollback_command_failed") {
    return "回滚/命令失败";
  }
  if (normalized === "rollback_broken_candidate") {
    return "回滚/候选跑坏了";
  }
  if (normalized === "rollback_scope_violation") {
    return "回滚/越界";
  }
  if (normalized === "rollback_irrelevant_change") {
    return "回滚/改动不相关";
  }
  if (normalized === "smoke_not_better") {
    return "快速比较没通过";
  }
  if (normalized === "codex_failed") {
    return "编辑代理失败";
  }
  return decision || "未标注结果";
}

function pickLatestMetricValue(finalMetrics: MetricSnapshot | null, smokeMetrics: MetricSnapshot | null, field: keyof MetricSnapshot): number | null {
  const finalValue = finalMetrics?.[field];
  if (typeof finalValue === "number" && Number.isFinite(finalValue)) {
    return finalValue;
  }
  const smokeValue = smokeMetrics?.[field];
  if (typeof smokeValue === "number" && Number.isFinite(smokeValue)) {
    return smokeValue;
  }
  return null;
}

function buildTrackResultSummary(
  track: CampaignTrack,
  {
    decision,
    smokeMetrics,
    finalMetrics,
  }: {
    decision: string;
    smokeMetrics: MetricSnapshot | null;
    finalMetrics: MetricSnapshot | null;
  },
): string {
  const seriesClass = inferTrackSeriesClass(track.trackId);
  const algorithmLabel = humanizeTrackModelFamily(track.runnerFamily ?? finalMetrics?.model_family ?? smokeMetrics?.model_family);
  const methodVariantLabel = inferTrackMethodVariantLabel(track.trackId, seriesClass);
  const statusLabel = humanizeTrackDecision(decision, isPromotableSeriesClass(seriesClass));
  const latestVal = pickLatestMetricValue(finalMetrics, smokeMetrics, "val_primary_metric");
  const latestRmse = pickLatestMetricValue(finalMetrics, smokeMetrics, "val_rmse");
  const parts = [`${methodVariantLabel} + ${algorithmLabel}`, statusLabel];
  if (latestVal !== null) {
    parts.push(`val r ${latestVal.toFixed(4)}`);
  }
  if (latestRmse !== null) {
    parts.push(`val RMSE ${latestRmse.toFixed(3)}`);
  }
  return parts.join(" · ");
}

export function applyTrackRuntimeResultSummary({
  trackState,
  track,
  runId,
  smokeMetrics,
  finalMetrics,
  decision,
}: {
  trackState: TrackRuntimeState;
  track: CampaignTrack;
  runId: string;
  smokeMetrics: MetricSnapshot | null;
  finalMetrics: MetricSnapshot | null;
  decision: string;
}): void {
  const seriesClass = inferTrackSeriesClass(track.trackId);
  const promotable = isPromotableSeriesClass(seriesClass);
  const latestValPrimary = pickLatestMetricValue(finalMetrics, smokeMetrics, "val_primary_metric");
  const latestValRmse = pickLatestMetricValue(finalMetrics, smokeMetrics, "val_rmse");
  const latestTestPrimary = pickLatestMetricValue(finalMetrics, smokeMetrics, "test_primary_metric");
  const latestTestRmse = pickLatestMetricValue(finalMetrics, smokeMetrics, "test_rmse");

  trackState.latest_run_id = runId;
  trackState.latest_smoke_run_id = smokeMetrics ? runId : "";
  trackState.latest_formal_run_id = finalMetrics ? runId : "";
  trackState.latest_val_primary_metric = latestValPrimary;
  trackState.latest_test_primary_metric = latestTestPrimary;
  trackState.latest_val_rmse = latestValRmse;
  trackState.latest_test_rmse = latestTestRmse;
  trackState.method_variant_label = inferTrackMethodVariantLabel(track.trackId, seriesClass);
  trackState.input_mode_label = inferTrackInputModeLabel(track.trackId, seriesClass);
  trackState.series_class = seriesClass;
  trackState.promotable = promotable;
  trackState.last_result_summary = buildTrackResultSummary(track, {
    decision,
    smokeMetrics,
    finalMetrics,
  });

  if (isBetterThan(latestValPrimary, trackState.best_val_primary_metric)) {
    trackState.best_val_primary_metric = latestValPrimary;
    trackState.best_test_primary_metric = latestTestPrimary;
    trackState.best_val_rmse = latestValRmse;
    trackState.best_test_rmse = latestTestRmse;
  }
}

export function hydrateTrackRuntimeState(
  trackState: TrackRuntimeState,
  track: CampaignTrack,
): TrackRuntimeState {
  const seriesClass = inferTrackSeriesClass(track.trackId);
  const resolvedSeriesClass = trackState.series_class || seriesClass;
  const hasLegacySeriesClass = trackState.series_class.trim() !== "";
  const runtimeTrack = track as RuntimeCampaignTrack;
  return {
    ...trackState,
    track_id: track.trackId,
    track_goal: track.trackGoal,
    promotion_target: track.promotionTarget,
    smoke_command: track.smokeCommand,
    formal_command: track.formalCommand,
    allowed_change_scope: [...track.allowedChangeScope],
    track_origin: runtimeTrack.trackOrigin ?? trackState.track_origin ?? "default",
    force_fresh_thread: runtimeTrack.forceFreshThread ?? trackState.force_fresh_thread ?? false,
    method_variant_label: trackState.method_variant_label || inferTrackMethodVariantLabel(track.trackId, resolvedSeriesClass),
    input_mode_label: trackState.input_mode_label || inferTrackInputModeLabel(track.trackId, resolvedSeriesClass),
    series_class: resolvedSeriesClass,
    promotable: hasLegacySeriesClass ? trackState.promotable : isPromotableSeriesClass(seriesClass),
  };
}

export function shouldForceFreshThread(track: Pick<RuntimeCampaignTrack, "trackOrigin" | "forceFreshThread">): boolean {
  return track.trackOrigin === "incubation" || Boolean(track.forceFreshThread);
}

export function summarizeThreadItemUsage(items: ThreadItem[]): ThreadItemUsageSummary {
  const summary: ThreadItemUsageSummary = {
    total_items: items.length,
    reasoning_items: 0,
    agent_messages: 0,
    command_executions: 0,
    file_changes: 0,
    web_searches: 0,
    mcp_tool_calls: 0,
    todo_lists: 0,
    errors: 0,
    completed_items: 0,
    failed_items: 0,
  };

  for (const item of items) {
    switch (item.type) {
      case "reasoning":
        summary.reasoning_items += 1;
        summary.completed_items += 1;
        break;
      case "agent_message":
        summary.agent_messages += 1;
        summary.completed_items += 1;
        break;
      case "command_execution":
        summary.command_executions += 1;
        if (item.status === "completed") {
          summary.completed_items += 1;
        } else if (item.status === "failed") {
          summary.failed_items += 1;
        }
        break;
      case "file_change":
        summary.file_changes += 1;
        if (item.status === "completed") {
          summary.completed_items += 1;
        } else if (item.status === "failed") {
          summary.failed_items += 1;
        }
        break;
      case "web_search":
        summary.web_searches += 1;
        break;
      case "mcp_tool_call":
        summary.mcp_tool_calls += 1;
        if (item.status === "failed") {
          summary.failed_items += 1;
        }
        break;
      case "todo_list":
        summary.todo_lists += 1;
        break;
      case "error":
        summary.errors += 1;
        break;
      default:
        break;
    }
  }

  return summary;
}

export function deriveTurnTelemetry({
  existing,
  items,
  decision,
  smokeMetrics,
  finalMetrics,
  nowIso,
}: {
  existing: TurnTelemetryState;
  items: ThreadItem[];
  decision: string;
  smokeMetrics: MetricSnapshot | null;
  finalMetrics: MetricSnapshot | null;
  nowIso: string;
}): TurnTelemetryState {
  const summary = summarizeThreadItemUsage(items);
  const sawSearch = summary.web_searches > 0 || summary.mcp_tool_calls > 0;
  const sawMaterialization = summary.command_executions > 0 || summary.file_changes > 0 || Boolean(smokeMetrics) || Boolean(finalMetrics);
  const staleReasonCodes = new Set(existing.stale_reason_codes ?? []);
  const pivotReasonCodes = new Set(existing.pivot_reason_codes ?? []);

  if (sawSearch && !sawMaterialization) {
    staleReasonCodes.add("search_only_no_materialization");
  }
  if (!smokeMetrics && sawMaterialization) {
    staleReasonCodes.add("no_new_smoke");
  }
  if (summary.web_searches >= 8 || summary.mcp_tool_calls >= 40) {
    staleReasonCodes.add("budget_capped");
  }
  if (decision === "smoke_not_better" || decision.startsWith("rollback")) {
    pivotReasonCodes.add("needs_new_direction");
  }
  if (finalMetrics && smokeMetrics) {
    pivotReasonCodes.add("formal_followup_available");
  }

  return {
    tool_usage_summary: summary,
    thinking_heartbeat_at: nowIso,
    last_retrieval_at: sawSearch ? nowIso : existing.last_retrieval_at,
    last_decision_at: nowIso,
    last_judgment_at: nowIso,
    last_materialization_at: sawMaterialization ? nowIso : existing.last_materialization_at,
    last_smoke_at: summary.command_executions > 0 || Boolean(smokeMetrics) ? nowIso : existing.last_smoke_at,
    stale_reason_codes: Array.from(staleReasonCodes),
    pivot_reason_codes: Array.from(pivotReasonCodes),
    search_budget_state: summary.web_searches >= 8 || summary.mcp_tool_calls >= 40
      ? "capped"
      : summary.web_searches >= 4 || summary.mcp_tool_calls >= 12
        ? "nearing_cap"
        : "healthy",
  };
}

export async function main() {
  const args = parseArgs(process.argv.slice(2));
  const campaignId = String(args["campaign-id"] ?? args["campaign_id"] ?? `autoresearch-campaign-${Date.now()}-${randomUUID().slice(0, 8)}`);
  const maxIterations = parsePositiveInt(args["max-iterations"] ?? args["max_iterations"], 6);
  const patience = parsePositiveInt(args["patience"], 3);
  const allowedDirs = parseAllowedDirs(args["allowed-dir"] ?? args["allowed_dirs"]);
  const baselineMetricsPath = resolveRepoPath(
    String(args["baseline-metrics-path"] ?? args["baseline_metrics_path"] ?? DEFAULT_BASELINE_METRICS_PATH),
  );
  const baselineCommand = String(args["baseline-command"] ?? args["baseline_command"] ?? DEFAULT_BASELINE_COMMAND);
  const smokeCommand = String(args["smoke-command"] ?? args["smoke_command"] ?? DEFAULT_SMOKE_COMMAND);
  const formalCommand = String(args["formal-command"] ?? args["formal_command"] ?? DEFAULT_FORMAL_COMMAND);
  const bankQcCommand = String(args["bank-qc-command"] ?? args["bank_qc_command"] ?? DEFAULT_BANK_QC_COMMAND);
  const runtimeOverlayInput = optionalString(
    args["runtime-track-overlay"] ?? args["runtime_track_overlay"] ?? process.env.AUTORESEARCH_RUNTIME_TRACK_OVERLAY,
  );
  const trackManifestPath = resolveRepoPath(String(args["track-manifest"] ?? args["track_manifest"] ?? DEFAULT_TRACK_MANIFEST_PATH));
  const smokeOutputDir = resolveRepoPath(String(args["smoke-output-dir"] ?? args["smoke_output_dir"] ?? DEFAULT_SMOKE_OUTPUT_DIR));
  const formalOutputDir = resolveRepoPath(String(args["formal-output-dir"] ?? args["formal_output_dir"] ?? DEFAULT_SMOKE_OUTPUT_DIR));
  const reviewPacketDir = resolveRepoPath(String(args["review-packet-dir"] ?? args["review_packet_dir"] ?? DEFAULT_REVIEW_PACKET_DIR));
  const dryRun = parseBoolean(args["dry-run"] ?? args["dry_run"]);

  await mkdir(path.dirname(STATUS_PATH), { recursive: true });
  await mkdir(TOOLS_ROOT, { recursive: true });
  await mkdir(path.dirname(TOOLS_LEDGER_PATH), { recursive: true });
  await mkdir(path.dirname(MONITOR_LEDGER_PATH), { recursive: true });
  await mkdir(smokeOutputDir, { recursive: true });
  await mkdir(formalOutputDir, { recursive: true });
  await mkdir(reviewPacketDir, { recursive: true });

  const programDocuments = await loadProgramDocuments(TOOLS_ROOT);
  const trackManifest = await loadTrackManifest(trackManifestPath, {
    defaultAllowedChangeScope: allowedDirs.map((dir) => normalizeRepoPath(dir)),
  });
  const runtimeOverlay = await loadRuntimeCampaignOverlay({
    repoRoot: REPO_ROOT,
    campaignId,
    overlayInput: runtimeOverlayInput,
  });
  const appliedTracks = applyRuntimeCampaignOverlay(trackManifest.tracks, runtimeOverlay);
  const effectiveTrackManifest = {
    ...trackManifest,
    tracks: appliedTracks.tracks,
  };

  const existingStatus = await readJsonIfExists<Partial<CampaignStatus>>(STATUS_PATH);
  const status = buildInitialStatus({
    campaignId,
    maxIterations,
    patience,
    existingStatus,
    baselineMetricsPath,
    allowedDirs,
    tracks: effectiveTrackManifest.tracks,
  });

  if (!existingStatus || existingStatus.campaign_id !== campaignId) {
    status.stage = "baseline";
    status.current_iteration = 0;
    status.current_command = "";
    status.patience_streak = 0;
    status.last_error = null;
    status.codex_thread_id = null;
    await writeStatus(status);
  }

  const bankQcReady = await runBankQcGate({
    status,
    campaignId,
    runId: `${campaignId}-bank-qc-bootstrap`,
    iteration: 0,
    parentRunId: null,
    track: null,
    bankQcCommand,
    reviewPacketDir,
    dryRun,
  });
  if (!bankQcReady) {
    return;
  }

  if (status.accepted_stable_best.run_id === "") {
    await initializeBaseline({
      status,
      baselineMetricsPath,
      baselineCommand,
      trackManifest: effectiveTrackManifest,
      reviewPacketDir,
      dryRun,
    });
  }

  if (status.current_iteration >= status.max_iterations) {
    status.stage = "done";
    status.current_command = "";
    status.updated_at = new Date().toISOString();
    await writeStatus(status);
    return;
  }

  if (dryRun) {
    status.stage = "paused";
    status.current_command = "";
    status.updated_at = new Date().toISOString();
    await writeStatus(status);
    return;
  }

  const consecutiveLimit = patience > 0 ? patience : Number.POSITIVE_INFINITY;
  while (hasRunnableTracks(status, consecutiveLimit)) {
    refreshCampaignControlState(status);
    if (status.budget_state === "capped") {
      break;
    }
    const codex = buildCampaignCodex(status.campaign_mode);
    let ranTrack = false;

    for (const track of effectiveTrackManifest.tracks) {
      const trackState = getTrackStateOrThrow(status, track.trackId);
      if (!canTrackStartNewEditTurn(trackState, {
        maxIterations: status.max_iterations,
        consecutiveLimit,
        maxEditTurnsPerTrack: MAX_EDIT_TURNS_PER_TRACK,
        noImprovementLimit: NO_IMPROVEMENT_LIMIT,
      })) {
        continue;
      }

      ranTrack = true;
      const trackAllowedDirs = resolveTrackAllowedDirs(track);
      const skipCodexEdit = shouldTrackSkipCodexEdit(track);
      const runtimeTrack = track as RuntimeCampaignTrack;
      const forceFreshThread = shouldForceFreshThread(runtimeTrack);
      const thread = skipCodexEdit
        ? null
        : forceFreshThread || !trackState.codex_thread_id
          ? codex.startThread({
              workingDirectory: REPO_ROOT,
              skipGitRepoCheck: true,
              sandboxMode: "workspace-write",
              approvalPolicy: "never",
              networkAccessEnabled: Boolean(track.internetResearchEnabled),
              additionalDirectories: trackAllowedDirs,
            })
          : codex.resumeThread(trackState.codex_thread_id, {
              workingDirectory: REPO_ROOT,
              skipGitRepoCheck: true,
              sandboxMode: "workspace-write",
              approvalPolicy: "never",
              networkAccessEnabled: Boolean(track.internetResearchEnabled),
              additionalDirectories: trackAllowedDirs,
            });

      const parentRunId = trackState.local_best.run_id || status.accepted_stable_best.run_id || null;
      const iteration = trackState.current_iteration + 1;
      const runId = buildTrackRunId(campaignId, track.trackId, iteration);
      const smokeOutputPath = buildTrackOutputPath(smokeOutputDir, track.trackId, `${runId}_smoke.json`);
      const formalOutputPath = buildTrackOutputPath(formalOutputDir, track.trackId, `${runId}_formal.json`);

      const iterationBankQcReady = await runBankQcGate({
        status,
        campaignId,
        runId: `${runId}-bank-qc`,
        iteration,
        parentRunId,
        track,
        bankQcCommand,
        reviewPacketDir,
        dryRun,
      });
      if (!iterationBankQcReady) {
        return;
      }

      const snapshot = await snapshotPaths(trackAllowedDirs);
      status.active_track_id = track.trackId;
      status.stage = skipCodexEdit ? "smoke" : "editing";
      status.current_command = skipCodexEdit ? "validated:skip_codex_edit" : "codex:edit";
      status.last_error = null;
      status.candidate = {
        ...buildEmptyCandidate(runId),
        stage: skipCodexEdit ? "smoke" : "editing",
        track_id: track.trackId,
        track_goal: track.trackGoal,
        promotion_target: track.promotionTarget,
        decision: skipCodexEdit ? "validated_preflighted" : "editing",
        next_step: skipCodexEdit ? "这条轨已通过预验证，直接进入 smoke。" : "等待候选改动生成。",
      };
      trackState.stage = skipCodexEdit ? "smoke" : "editing";
      trackState.last_run_id = runId;
      trackState.updated_at = new Date().toISOString();
      status.updated_at = new Date().toISOString();
      await writeStatus(status);

      let codexResult: CodexEditResult;
      let touchedFiles: string[] = [];
      if (skipCodexEdit) {
        codexResult = buildValidatedTrackBypassResult(track);
      } else {
        try {
          codexResult = await runCodexEditTurn({
            thread: thread!,
            campaignId,
            iteration,
            status,
            track,
            allowedDirs: trackAllowedDirs,
            programDocuments,
          });
          trackState.codex_thread_id = codexResult.threadId;
        } catch (error) {
          await restoreSnapshot(snapshot, touchedFiles);
          status.stage = "paused";
          status.current_command = "";
          status.last_error = stringifyError(error);
          status.candidate = {
            ...buildEmptyCandidate(runId),
            track_id: track.trackId,
            track_goal: track.trackGoal,
            promotion_target: track.promotionTarget,
            rollback_applied: true,
            allowed_scope_ok: false,
            decision: "codex_failed",
            next_step: "修复代理提示或命令，再继续下一轮。",
            artifacts: [STATUS_PATH, TOOLS_LEDGER_PATH, MONITOR_LEDGER_PATH],
          };
          trackState.stage = "paused";
          trackState.last_decision = "codex_failed";
          trackState.patience_streak += 1;
          trackState.current_iteration = iteration;
          applyTrackRuntimeResultSummary({
            trackState,
            track,
            runId,
            smokeMetrics: null,
            finalMetrics: null,
            decision: "codex_failed",
          });
          trackState.updated_at = new Date().toISOString();
          status.updated_at = new Date().toISOString();
          const failureTelemetry = deriveTurnTelemetry({
            existing: {
              tool_usage_summary: status.tool_usage_summary,
              thinking_heartbeat_at: status.thinking_heartbeat_at,
              last_retrieval_at: status.last_retrieval_at,
              last_decision_at: status.last_decision_at,
              last_judgment_at: status.last_judgment_at,
              last_materialization_at: status.last_materialization_at,
              last_smoke_at: status.last_smoke_at,
              stale_reason_codes: status.stale_reason_codes,
              pivot_reason_codes: status.pivot_reason_codes,
              search_budget_state: status.search_budget_state,
            },
            items: [],
            decision: "codex_failed",
            smokeMetrics: null,
            finalMetrics: null,
            nowIso: new Date().toISOString(),
          });
          await recordIteration({
            status,
            iteration,
            runId,
            parentRunId,
            track,
            smokeMetrics: null,
            finalMetrics: null,
            filesTouched: [],
            commands: [],
            searchQueries: [],
            researchEvidence: [],
            allowedScopeOk: false,
            rollbackApplied: true,
            relevanceLabel: "off_track_but_ran",
            relevanceReason: "代理回合在生成候选前就失败了，所以这轮没有形成可比较的实验改动。",
            decision: "codex_failed",
            nextStep: status.candidate.next_step,
            hypothesis: status.candidate.hypothesis,
            whyThisChange: status.candidate.why_this_change,
            changesSummary: status.candidate.changes_summary,
            changeBucket: status.candidate.change_bucket,
            trackComparisonNote: status.candidate.track_comparison_note,
            artifacts: status.candidate.artifacts,
            telemetry: failureTelemetry,
            reviewPacketDir,
          });
          await writeStatus(status);
          continue;
        }
      }

      const audit = skipCodexEdit
        ? buildValidatedTrackAudit()
        : auditFileChanges(codexResult.items, trackAllowedDirs);
      const relevance = skipCodexEdit
        ? buildValidatedTrackRelevance()
        : classifyCommandRelevance(audit.filesTouched, track.smokeCommand, track.formalCommand);
      touchedFiles = audit.filesTouched;
      const observedCommands = collectCommandStrings(codexResult.items);
      const proposal = codexResult.proposal;

      let smokeMetrics: MetricSnapshot | null = null;
      let finalMetrics: MetricSnapshot | null = null;
      let rollbackApplied = false;
      let allowedScopeOk = audit.allowedScopeOk;
      let decision = "reject";
      let nextStep = "换一个更小的改动，再跑 smoke。";
      let candidateArtifacts = [STATUS_PATH, TOOLS_LEDGER_PATH, MONITOR_LEDGER_PATH];
      let relevanceLabel = relevance.label;
      let relevanceReason = relevance.reason;

      if (!allowedScopeOk) {
        await restoreSnapshot(snapshot, touchedFiles);
        rollbackApplied = true;
        decision = "rollback_scope_violation";
        nextStep = "这轮碰到了禁区或越出了允许目录，先撤回后再重试。";
        status.last_error = audit.violations.join(", ");
      } else if (!dryRun) {
        const smokeCommandLine = ensureOutputJson(track.smokeCommand, smokeOutputPath);
        status.stage = "smoke";
        status.current_command = smokeCommandLine;
        trackState.stage = "smoke";
        status.updated_at = new Date().toISOString();
        await writeStatus(status);

        const smokeRun = await runShellCommand(smokeCommandLine, REPO_ROOT, {
          campaignId: status.campaign_id,
          trackId: track.trackId,
          taskKind: "smoke_train",
          modelFamily: track.runnerFamily ?? null,
          priority: 2,
          expectedMemoryClass: track.expectedMemoryClass ?? "low",
        });
        const smokeMetricsPayload = await readJsonIfExists<Record<string, unknown>>(smokeOutputPath);
        smokeMetrics = summarizeMetrics(smokeMetricsPayload, smokeOutputPath);
        candidateArtifacts = dedupeStrings([
          smokeOutputPath,
          smokeMetrics?.best_checkpoint_path ?? undefined,
          smokeMetrics?.last_checkpoint_path ?? undefined,
          STATUS_PATH,
          TOOLS_LEDGER_PATH,
          MONITOR_LEDGER_PATH,
        ]);
        observedCommands.push(smokeRun.command);

        if (smokeRun.exitCode !== 0 || !smokeMetrics || smokeMetrics.val_primary_metric === null) {
          await restoreSnapshot(snapshot, touchedFiles);
          rollbackApplied = true;
          decision = smokeRun.exitCode === 0 ? "rollback_broken_candidate" : "rollback_command_failed";
          nextStep = "先把 smoke 命令跑通，再继续。";
          status.last_error = smokeRun.exitCode === 0 ? "smoke metrics missing" : (smokeRun.stderr.trim() || smokeRun.stdout.trim() || "smoke command failed");
        } else if (isBetterThan(smokeMetrics.val_primary_metric, trackState.local_best.val_primary_metric)) {
          status.stage = "formal_eval";
          const formalCommandLine = ensureOutputJson(track.formalCommand, formalOutputPath);
          status.current_command = formalCommandLine;
          trackState.stage = "formal_eval";
          status.updated_at = new Date().toISOString();
          await writeStatus(status);

          const formalRun = await runShellCommand(formalCommandLine, REPO_ROOT, {
            campaignId: status.campaign_id,
            trackId: track.trackId,
            taskKind: "formal_train",
            modelFamily: track.runnerFamily ?? null,
            priority: 1,
            expectedMemoryClass: track.expectedMemoryClass ?? "low",
          });
          const formalMetricsPayload = await readJsonIfExists<Record<string, unknown>>(formalOutputPath);
          finalMetrics = summarizeMetrics(formalMetricsPayload, formalOutputPath);
          observedCommands.push(formalRun.command);
          candidateArtifacts = dedupeStrings([
            ...candidateArtifacts,
            formalOutputPath,
            finalMetrics?.best_checkpoint_path ?? undefined,
            finalMetrics?.last_checkpoint_path ?? undefined,
          ]);

          if (formalRun.exitCode !== 0 || !finalMetrics || finalMetrics.val_primary_metric === null) {
            await restoreSnapshot(snapshot, touchedFiles);
            rollbackApplied = true;
            decision = formalRun.exitCode === 0 ? "rollback_broken_candidate" : "rollback_command_failed";
            nextStep = "formal 没过，回到 smoke 再换一个候选。";
            status.last_error = formalRun.exitCode === 0 ? "formal metrics missing" : (formalRun.stderr.trim() || formalRun.stdout.trim() || "formal command failed");
          } else {
            await restoreSnapshot(snapshot, touchedFiles);
            rollbackApplied = true;
            const outcome = decideTrackOutcome({
              promotionTarget: track.promotionTarget,
              formalValMetric: finalMetrics.val_primary_metric,
              localBestMetric: trackState.local_best.val_primary_metric,
              globalStableBestMetric: status.accepted_stable_best.val_primary_metric,
              canonicalRetestReady: isCanonicalPromotionReady(track, finalMetrics),
            });
            decision = outcome.decision;
            if (outcome.updateLocalBest) {
              trackState.local_best = buildAcceptedBest({
                runId,
                primaryMetricName: smokeMetrics.primary_metric_name,
                smokeMetrics,
                formalMetrics: finalMetrics,
              });
            }
            if (outcome.updateGlobalCandidate) {
              status.leading_unverified_candidate = buildAcceptedBest({
                runId,
                primaryMetricName: smokeMetrics.primary_metric_name,
                smokeMetrics,
                formalMetrics: finalMetrics,
              });
              syncAcceptedBestAlias(status);
              nextStep = "canonical mainline formal 超过 stable best，先记成未复验候选，再进 daily packet。";
            } else if (decision === "hold_for_promotion_review") {
              nextStep = "track 局部得分提升，等待 daily packet 决定是否做 canonical retest。";
            } else {
              nextStep = "formal 没超过当前 local best，换更小的改动。";
            }
          }
        } else {
          await restoreSnapshot(snapshot, touchedFiles);
          rollbackApplied = true;
          decision = "smoke_not_better";
          nextStep = "先换方向，不要继续放大这个候选。";
        }
      } else {
        rollbackApplied = true;
        nextStep = "dry-run 只做了文件审计，没有跑校验。";
      }

      if (decision === "hold_for_packet_gate" || decision === "hold_for_promotion_review") {
        trackState.patience_streak = 0;
        trackState.stage = decision === "hold_for_packet_gate" ? "formal_eval" : "accepted";
        status.stage = trackState.stage;
        status.last_error = null;
      } else {
        trackState.patience_streak += 1;
        trackState.stage = rollbackApplied ? "rollback" : "rejected";
        status.stage = trackState.stage;
      }

      trackState.current_iteration = iteration;
      if (!skipCodexEdit) {
        trackState.edit_turns_used += 1;
      }
      if (finalMetrics?.val_primary_metric !== null) {
        trackState.formal_runs_completed += 1;
      }
      trackState.last_run_id = runId;
      trackState.last_decision = decision;
      applyTrackRuntimeResultSummary({
        trackState,
        track,
        runId,
        smokeMetrics,
        finalMetrics,
        decision,
      });
      const telemetry = deriveTurnTelemetry({
        existing: {
          tool_usage_summary: status.tool_usage_summary,
          thinking_heartbeat_at: status.thinking_heartbeat_at,
          last_retrieval_at: status.last_retrieval_at,
          last_decision_at: status.last_decision_at,
          last_judgment_at: status.last_judgment_at,
          last_materialization_at: status.last_materialization_at,
          last_smoke_at: status.last_smoke_at,
          stale_reason_codes: status.stale_reason_codes,
          pivot_reason_codes: status.pivot_reason_codes,
          search_budget_state: status.search_budget_state,
        },
        items: codexResult.items,
        decision,
        smokeMetrics,
        finalMetrics,
        nowIso: new Date().toISOString(),
      });
      status.tool_usage_summary = telemetry.tool_usage_summary;
      status.thinking_heartbeat_at = telemetry.thinking_heartbeat_at;
      status.last_retrieval_at = telemetry.last_retrieval_at;
      status.last_decision_at = telemetry.last_decision_at;
      status.last_judgment_at = telemetry.last_judgment_at;
      status.last_materialization_at = telemetry.last_materialization_at;
      status.last_smoke_at = telemetry.last_smoke_at;
      status.stale_reason_codes = telemetry.stale_reason_codes;
      status.pivot_reason_codes = telemetry.pivot_reason_codes;
      status.search_budget_state = telemetry.search_budget_state;
      trackState.tool_usage_summary = telemetry.tool_usage_summary;
      trackState.thinking_heartbeat_at = telemetry.thinking_heartbeat_at;
      trackState.last_retrieval_at = telemetry.last_retrieval_at;
      trackState.last_decision_at = telemetry.last_decision_at;
      trackState.last_judgment_at = telemetry.last_judgment_at;
      trackState.last_materialization_at = telemetry.last_materialization_at;
      trackState.last_smoke_at = telemetry.last_smoke_at;
      trackState.stale_reason_codes = telemetry.stale_reason_codes;
      trackState.pivot_reason_codes = telemetry.pivot_reason_codes;
      trackState.search_budget_state = telemetry.search_budget_state;
      trackState.updated_at = new Date().toISOString();
      status.current_command = "";
      status.candidate = {
        run_id: runId,
        stage: decision === "hold_for_packet_gate"
          ? "formal_eval"
          : decision === "hold_for_promotion_review"
            ? "accepted"
          : rollbackApplied
            ? "rollback"
            : finalMetrics
              ? "formal_eval"
              : smokeMetrics
                ? "smoke"
                : "baseline",
        track_id: track.trackId,
        track_goal: track.trackGoal,
        promotion_target: track.promotionTarget,
        hypothesis: proposal.hypothesis,
        why_this_change: proposal.why_this_change,
        changes_summary: proposal.changes_summary,
        change_bucket: proposal.change_bucket,
        track_comparison_note: proposal.track_comparison_note,
        files_touched: touchedFiles,
        commands: dedupeStrings(observedCommands),
        search_queries: proposal.search_queries,
        research_evidence: proposal.research_evidence,
        smoke_metrics: smokeMetrics,
        final_metrics: finalMetrics,
        allowed_scope_ok: allowedScopeOk,
        rollback_applied: rollbackApplied,
        relevance_label: relevanceLabel,
        relevance_reason: relevanceReason,
        decision,
        next_step: nextStep,
        artifacts: candidateArtifacts,
        tool_usage_summary: telemetry.tool_usage_summary,
        thinking_heartbeat_at: telemetry.thinking_heartbeat_at,
        last_retrieval_at: telemetry.last_retrieval_at,
        last_decision_at: telemetry.last_decision_at,
        last_judgment_at: telemetry.last_judgment_at,
        last_materialization_at: telemetry.last_materialization_at,
        last_smoke_at: telemetry.last_smoke_at,
        stale_reason_codes: telemetry.stale_reason_codes,
        pivot_reason_codes: telemetry.pivot_reason_codes,
        search_budget_state: telemetry.search_budget_state,
      };
      status.updated_at = new Date().toISOString();
      await recordIteration({
        status,
        iteration,
        runId,
        parentRunId,
        track,
        smokeMetrics,
        finalMetrics,
        filesTouched: touchedFiles,
        commands: dedupeStrings(observedCommands),
        searchQueries: proposal.search_queries,
        researchEvidence: proposal.research_evidence,
        allowedScopeOk,
        rollbackApplied,
        relevanceLabel,
        relevanceReason,
        decision,
        nextStep,
        hypothesis: proposal.hypothesis,
        whyThisChange: proposal.why_this_change,
        changesSummary: proposal.changes_summary,
        changeBucket: proposal.change_bucket,
        trackComparisonNote: proposal.track_comparison_note,
        artifacts: candidateArtifacts,
        telemetry,
        reviewPacketDir,
      });
      await writeStatus(status);
    }

    if (!ranTrack) {
      break;
    }
  }

  status.stage = "done";
  status.current_command = "";
  if (status.stop_reason !== "manual_stop_loss") {
    const control = deriveCampaignControlState({ status });
    status.stop_reason = control.stopReason;
  }
  status.updated_at = new Date().toISOString();
  await writeStatus(status);
}

async function initializeBaseline({
  status,
  baselineMetricsPath,
  baselineCommand,
  trackManifest,
  reviewPacketDir,
  dryRun,
}: {
  status: CampaignStatus;
  baselineMetricsPath: string;
  baselineCommand: string;
  trackManifest: { tracks: CampaignTrack[] };
  reviewPacketDir: string;
  dryRun: boolean;
}) {
  let baselineMetrics = await readJsonIfExists<Record<string, unknown>>(baselineMetricsPath);
  let baselineArtifacts = [baselineMetricsPath, STATUS_PATH, TOOLS_LEDGER_PATH, MONITOR_LEDGER_PATH];
  let baselineCommandUsed = "";

  if ((!baselineMetrics || !isFiniteNumber(extractPrimaryMetric(baselineMetrics))) && baselineCommand && !dryRun) {
    status.stage = "baseline";
    status.current_command = "baseline";
    status.updated_at = new Date().toISOString();
    await writeStatus(status);

    const outputPath = baselineMetricsPath;
    const result = await runShellCommand(ensureOutputJson(baselineCommand, outputPath), REPO_ROOT, {
      campaignId: status.campaign_id,
      trackId: "canonical_mainline",
      taskKind: "baseline_train",
      modelFamily: "baseline",
      priority: 2,
      expectedMemoryClass: "low",
    });
    baselineCommandUsed = result.command;
    if (result.exitCode !== 0) {
      throw new Error(`baseline command failed: ${result.command}`);
    }
    baselineMetrics = await readJsonIfExists<Record<string, unknown>>(baselineMetricsPath);
  }

  const summary = summarizeMetrics(baselineMetrics, baselineMetricsPath);
  if (!summary || summary.val_primary_metric === null) {
    throw new Error(`baseline metrics missing primary metric: ${baselineMetricsPath}`);
  }

  status.accepted_best = buildAcceptedBest({
    runId: `${status.campaign_id}-baseline`,
    primaryMetricName: summary.primary_metric_name,
    smokeMetrics: summary,
    formalMetrics: null,
  });
  status.frozen_baseline = { ...status.accepted_best };
  status.accepted_stable_best = { ...status.accepted_best };
  status.leading_unverified_candidate = normalizeAcceptedBest(null);
  syncAcceptedBestAlias(status);
  primeCanonicalTrackLocalBest(status, trackManifest.tracks);
  status.stage = "smoke";
  status.current_command = baselineCommandUsed;
  status.candidate = buildEmptyCandidate(`${status.campaign_id}-baseline`);
  status.candidate.stage = "baseline";
  status.candidate.track_id = "canonical_mainline";
  status.candidate.track_goal = "Protect the canonical joints mainline.";
  status.candidate.promotion_target = "canonical_mainline";
  status.candidate.smoke_metrics = summary;
  status.candidate.final_metrics = null;
  status.candidate.allowed_scope_ok = true;
  status.candidate.rollback_applied = false;
  status.candidate.decision = "baseline_initialized";
  status.candidate.next_step = "开始 smoke 候选。";
  status.candidate.track_comparison_note = "Canonical baseline initialized.";
  status.candidate.artifacts = baselineArtifacts;
  status.updated_at = new Date().toISOString();
  await writeStatus(status);

  const telemetry = deriveTurnTelemetry({
    existing: {
      tool_usage_summary: status.tool_usage_summary,
      thinking_heartbeat_at: status.thinking_heartbeat_at,
      last_retrieval_at: status.last_retrieval_at,
      last_decision_at: status.last_decision_at,
      last_judgment_at: status.last_judgment_at,
      last_materialization_at: status.last_materialization_at,
      last_smoke_at: status.last_smoke_at,
      stale_reason_codes: status.stale_reason_codes,
      pivot_reason_codes: status.pivot_reason_codes,
      search_budget_state: status.search_budget_state,
    },
    items: [],
    decision: "baseline_initialized",
    smokeMetrics: summary,
    finalMetrics: null,
    nowIso: new Date().toISOString(),
  });

  await recordIteration({
    status,
    iteration: 0,
    runId: status.candidate.run_id,
    parentRunId: null,
    track: findCanonicalTrack(trackManifest.tracks),
    smokeMetrics: summary,
    finalMetrics: null,
    filesTouched: [],
    commands: baselineCommandUsed ? [baselineCommandUsed] : [],
    searchQueries: [],
    researchEvidence: [],
    allowedScopeOk: true,
    rollbackApplied: false,
    relevanceLabel: "on_track",
    relevanceReason: "这是一条主线 baseline 写入记录，用来给后续正式比较提供固定参考线。",
    decision: "baseline_initialized",
    nextStep: "开始 smoke 候选。",
    hypothesis: "用已有 smoke 结果初始化 campaign 基线。",
    whyThisChange: "先把当前最好结果写入 monitor，再开始后续 smoke/formal 迭代。",
    changesSummary: "初始化 baseline，确认 campaign 可以从现有 smoke 指标继续跑。",
    changeBucket: "model-led",
    trackComparisonNote: "Canonical baseline established for future promotion comparisons.",
    artifacts: baselineArtifacts,
    telemetry,
    reviewPacketDir,
  });
}

async function runBankQcGate({
  status,
  campaignId,
  runId,
  iteration,
  parentRunId,
  track,
  bankQcCommand,
  reviewPacketDir,
  dryRun,
}: {
  status: CampaignStatus;
  campaignId: string;
  runId: string;
  iteration: number;
  parentRunId: string | null;
  track: CampaignTrack | null;
  bankQcCommand: string;
  reviewPacketDir: string;
  dryRun: boolean;
}): Promise<boolean> {
  if (!bankQcCommand) {
    return true;
  }
  if (dryRun) {
    return true;
  }

  status.stage = "bank_qc";
  status.current_command = bankQcCommand;
  status.last_error = null;
  status.candidate = {
    ...buildEmptyCandidate(runId),
    stage: "bank_qc",
    track_id: track?.trackId ?? "",
    track_goal: track?.trackGoal ?? "",
    promotion_target: track?.promotionTarget ?? "",
    decision: "bank_qc",
    next_step: "等待 bank-QC 结果。",
  };
  status.updated_at = new Date().toISOString();
  await writeStatus(status);

  const result = await runShellCommand(bankQcCommand, REPO_ROOT, {
    campaignId: status.campaign_id,
    trackId: null,
    taskKind: "bank_qc",
    modelFamily: "service",
    priority: 3,
    expectedMemoryClass: "service",
  });
  if (result.exitCode === 0) {
    status.current_command = "";
    status.updated_at = new Date().toISOString();
    await writeStatus(status);
    return true;
  }

  status.stage = "bank_qc_failed";
  status.current_command = "";
  status.last_error = result.stderr.trim() || result.stdout.trim() || "bank-QC gate failed";
  status.candidate = {
    ...buildEmptyCandidate(runId),
    stage: "bank_qc_failed",
    track_id: track?.trackId ?? "",
    track_goal: track?.trackGoal ?? "",
    promotion_target: track?.promotionTarget ?? "",
    hypothesis: "先确认当前 dataset 的 active_bank 和 raw channel half scan 一致。",
    why_this_change: "bank-QC 失败时，后续训练结果不再可比较。",
    changes_summary: "bank-QC 门禁失败，训练没有开始。",
    track_comparison_note: "Safety gate failed before any track-level comparison.",
    files_touched: [],
    commands: [result.command],
    smoke_metrics: null,
    final_metrics: null,
    allowed_scope_ok: true,
    rollback_applied: false,
    decision: "bank_qc_failed",
    next_step: "先修正 active_bank 或 channel scan，再重新启动 campaign。",
    artifacts: [STATUS_PATH, TOOLS_LEDGER_PATH, MONITOR_LEDGER_PATH],
  };
  status.updated_at = new Date().toISOString();
  const telemetry = deriveTurnTelemetry({
    existing: {
      tool_usage_summary: status.tool_usage_summary,
      thinking_heartbeat_at: status.thinking_heartbeat_at,
      last_retrieval_at: status.last_retrieval_at,
      last_decision_at: status.last_decision_at,
      last_judgment_at: status.last_judgment_at,
      last_materialization_at: status.last_materialization_at,
      last_smoke_at: status.last_smoke_at,
      stale_reason_codes: status.stale_reason_codes,
      pivot_reason_codes: status.pivot_reason_codes,
      search_budget_state: status.search_budget_state,
    },
    items: [],
    decision: "bank_qc_failed",
    smokeMetrics: null,
    finalMetrics: null,
    nowIso: new Date().toISOString(),
  });
  await recordIteration({
    status,
    iteration,
    runId,
    parentRunId,
    track,
    smokeMetrics: null,
    finalMetrics: null,
    filesTouched: [],
    commands: [result.command],
    searchQueries: [],
    researchEvidence: [],
    allowedScopeOk: true,
    rollbackApplied: false,
    relevanceLabel: "on_track",
    relevanceReason: "bank-QC 是这轮实验的硬安全门，必须先通过才能进入正式比较。",
    decision: "bank_qc_failed",
    nextStep: status.candidate.next_step,
    hypothesis: status.candidate.hypothesis,
    whyThisChange: status.candidate.why_this_change,
    changesSummary: status.candidate.changes_summary,
    changeBucket: status.candidate.change_bucket,
    trackComparisonNote: status.candidate.track_comparison_note,
    artifacts: status.candidate.artifacts,
    telemetry,
    reviewPacketDir,
  });
  await writeStatus(status);
  return false;
}

function buildInitialStatus({
  campaignId,
  maxIterations,
  patience,
  existingStatus,
  baselineMetricsPath,
  allowedDirs,
  tracks,
}: {
  campaignId: string;
  maxIterations: number;
  patience: number;
  existingStatus: Partial<CampaignStatus> | null;
  baselineMetricsPath: string;
  allowedDirs: string[];
  tracks: CampaignTrack[];
}): CampaignStatus {
  if (existingStatus && existingStatus.campaign_id === campaignId) {
    const activeTrackId =
      typeof existingStatus.active_track_id === "string" && tracks.some((track) => track.trackId === existingStatus.active_track_id)
        ? existingStatus.active_track_id
        : null;
    return {
      campaign_id: campaignId,
      started_at: typeof (existingStatus as Record<string, unknown>).started_at === "string"
        ? String((existingStatus as Record<string, unknown>).started_at)
        : new Date().toISOString(),
      current_iteration: Number(existingStatus.current_iteration ?? 0),
      max_iterations: Number(existingStatus.max_iterations ?? maxIterations),
      patience: Number(existingStatus.patience ?? patience),
      stage: normalizeStage(existingStatus.stage),
      active_track_id: activeTrackId,
      frozen_baseline: normalizeAcceptedBest(
        (existingStatus as Record<string, unknown>).frozen_baseline ?? existingStatus.accepted_best,
      ),
      accepted_stable_best: normalizeAcceptedBest(
        (existingStatus as Record<string, unknown>).accepted_stable_best ?? existingStatus.accepted_best,
      ),
      leading_unverified_candidate: normalizeAcceptedBest(
        (existingStatus as Record<string, unknown>).leading_unverified_candidate,
      ),
      accepted_best: normalizeAcceptedBest(
        (existingStatus as Record<string, unknown>).accepted_stable_best ?? existingStatus.accepted_best,
      ),
      candidate: normalizeCandidate(existingStatus.candidate),
      track_states: syncTrackStates(
        normalizeTrackStates((existingStatus as Record<string, unknown>).track_states),
        tracks,
      ),
      current_command: String(existingStatus.current_command ?? ""),
      updated_at: new Date().toISOString(),
      campaign_mode: normalizeCampaignMode((existingStatus as Record<string, unknown>).campaign_mode),
      budget_state: normalizeBudgetState((existingStatus as Record<string, unknown>).budget_state),
      stop_reason: normalizeStopReason((existingStatus as Record<string, unknown>).stop_reason),
      codex_thread_id: typeof existingStatus.codex_thread_id === "string" ? existingStatus.codex_thread_id : null,
      patience_streak: Number(existingStatus.patience_streak ?? 0),
      last_error: typeof existingStatus.last_error === "string" ? existingStatus.last_error : null,
      tool_usage_summary: normalizeThreadItemUsageSummary((existingStatus as Record<string, unknown>).tool_usage_summary),
      thinking_heartbeat_at: String((existingStatus as Record<string, unknown>).thinking_heartbeat_at ?? ""),
      last_retrieval_at: String((existingStatus as Record<string, unknown>).last_retrieval_at ?? ""),
      last_decision_at: String((existingStatus as Record<string, unknown>).last_decision_at ?? ""),
      last_judgment_at: String((existingStatus as Record<string, unknown>).last_judgment_at ?? ""),
      last_materialization_at: String((existingStatus as Record<string, unknown>).last_materialization_at ?? ""),
      last_smoke_at: String((existingStatus as Record<string, unknown>).last_smoke_at ?? ""),
      stale_reason_codes: parseStringArray((existingStatus as Record<string, unknown>).stale_reason_codes),
      pivot_reason_codes: parseStringArray((existingStatus as Record<string, unknown>).pivot_reason_codes),
      search_budget_state: normalizeBudgetState((existingStatus as Record<string, unknown>).search_budget_state),
    };
  }

  return {
    campaign_id: campaignId,
    started_at: new Date().toISOString(),
    current_iteration: 0,
    max_iterations: maxIterations,
    patience,
    stage: "baseline",
    active_track_id: null,
    frozen_baseline: normalizeAcceptedBest(null),
    accepted_stable_best: normalizeAcceptedBest(null),
    leading_unverified_candidate: normalizeAcceptedBest(null),
    accepted_best: {
      run_id: "",
      dataset_name: "",
      target_mode: "",
      target_space: "",
      primary_metric_name: "",
      val_primary_metric: null,
      formal_val_primary_metric: null,
      val_rmse: null,
      test_primary_metric: null,
      test_rmse: null,
      feature_family: null,
      model_family: null,
      evaluation_mode: null,
      result_json: null,
      artifacts: [],
    },
    candidate: buildEmptyCandidate(`${campaignId}-bootstrap`),
    track_states: tracks.map((track) => buildTrackRuntimeState(track)),
    current_command: "",
    updated_at: new Date().toISOString(),
    campaign_mode: "exploration",
    budget_state: "healthy",
    stop_reason: "none",
    codex_thread_id: null,
    patience_streak: 0,
    last_error: null,
    tool_usage_summary: null,
    thinking_heartbeat_at: "",
    last_retrieval_at: "",
    last_decision_at: "",
    last_judgment_at: "",
    last_materialization_at: "",
    last_smoke_at: "",
    stale_reason_codes: [],
    pivot_reason_codes: [],
    search_budget_state: "healthy",
  };
}

function normalizeTrackStates(value: unknown): TrackRuntimeState[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((record) => ({
      track_id: String(record.track_id ?? ""),
      track_goal: String(record.track_goal ?? ""),
      promotion_target: String(record.promotion_target ?? ""),
      smoke_command: String(record.smoke_command ?? ""),
      formal_command: String(record.formal_command ?? ""),
      allowed_change_scope: parseStringArray(record.allowed_change_scope),
      track_origin: normalizeTrackOrigin(record.track_origin),
      force_fresh_thread: parseBoolean(record.force_fresh_thread),
      current_iteration: Number(record.current_iteration ?? 0),
      patience_streak: Number(record.patience_streak ?? 0),
      edit_turns_used: Number(record.edit_turns_used ?? record.current_iteration ?? 0),
      formal_runs_completed: Number(record.formal_runs_completed ?? 0),
      stage: normalizeStage(record.stage),
      last_run_id: String(record.last_run_id ?? ""),
      last_decision: String(record.last_decision ?? ""),
      codex_thread_id: typeof record.codex_thread_id === "string" ? record.codex_thread_id : null,
      local_best: normalizeAcceptedBest(record.local_best),
      latest_run_id: String(record.latest_run_id ?? ""),
      latest_smoke_run_id: String(record.latest_smoke_run_id ?? ""),
      latest_formal_run_id: String(record.latest_formal_run_id ?? ""),
      latest_val_primary_metric: toNumberOrNull(record.latest_val_primary_metric),
      latest_test_primary_metric: toNumberOrNull(record.latest_test_primary_metric),
      latest_val_rmse: toNumberOrNull(record.latest_val_rmse),
      latest_test_rmse: toNumberOrNull(record.latest_test_rmse),
      best_val_primary_metric: toNumberOrNull(record.best_val_primary_metric),
      best_test_primary_metric: toNumberOrNull(record.best_test_primary_metric),
      best_val_rmse: toNumberOrNull(record.best_val_rmse),
      best_test_rmse: toNumberOrNull(record.best_test_rmse),
      last_result_summary: String(record.last_result_summary ?? ""),
      method_variant_label: String(record.method_variant_label ?? ""),
      input_mode_label: String(record.input_mode_label ?? ""),
      series_class: String(record.series_class ?? ""),
      promotable: parseBoolean(record.promotable),
      tool_usage_summary: normalizeThreadItemUsageSummary(record.tool_usage_summary),
      thinking_heartbeat_at: String(record.thinking_heartbeat_at ?? ""),
      last_retrieval_at: String(record.last_retrieval_at ?? ""),
      last_decision_at: String(record.last_decision_at ?? ""),
      last_judgment_at: String(record.last_judgment_at ?? ""),
      last_materialization_at: String(record.last_materialization_at ?? ""),
      last_smoke_at: String(record.last_smoke_at ?? ""),
      stale_reason_codes: parseStringArray(record.stale_reason_codes),
      pivot_reason_codes: parseStringArray(record.pivot_reason_codes),
      search_budget_state: normalizeBudgetState(record.search_budget_state),
      updated_at: String(record.updated_at ?? new Date().toISOString()),
    }));
}

function buildTrackRuntimeState(track: CampaignTrack): TrackRuntimeState {
  const seriesClass = inferTrackSeriesClass(track.trackId);
  const runtimeTrack = track as RuntimeCampaignTrack;
  return {
    track_id: track.trackId,
    track_goal: track.trackGoal,
    promotion_target: track.promotionTarget,
    smoke_command: track.smokeCommand,
    formal_command: track.formalCommand,
    allowed_change_scope: [...track.allowedChangeScope],
    track_origin: runtimeTrack.trackOrigin ?? "default",
    force_fresh_thread: runtimeTrack.forceFreshThread ?? false,
    current_iteration: 0,
    patience_streak: 0,
    edit_turns_used: 0,
    formal_runs_completed: 0,
    stage: "baseline",
    last_run_id: "",
    last_decision: "",
    codex_thread_id: null,
    local_best: normalizeAcceptedBest(null),
    latest_run_id: "",
    latest_smoke_run_id: "",
    latest_formal_run_id: "",
    latest_val_primary_metric: null,
    latest_test_primary_metric: null,
    latest_val_rmse: null,
    latest_test_rmse: null,
    best_val_primary_metric: null,
    best_test_primary_metric: null,
    best_val_rmse: null,
    best_test_rmse: null,
    last_result_summary: "",
    method_variant_label: inferTrackMethodVariantLabel(track.trackId, seriesClass),
    input_mode_label: inferTrackInputModeLabel(track.trackId, seriesClass),
    series_class: seriesClass,
    promotable: isPromotableSeriesClass(seriesClass),
    tool_usage_summary: null,
    thinking_heartbeat_at: "",
    last_retrieval_at: "",
    last_decision_at: "",
    last_judgment_at: "",
    last_materialization_at: "",
    last_smoke_at: "",
    stale_reason_codes: [],
    pivot_reason_codes: [],
    search_budget_state: "healthy",
    updated_at: new Date().toISOString(),
  };
}

function syncTrackStates(existing: TrackRuntimeState[], tracks: CampaignTrack[]): TrackRuntimeState[] {
  const byId = new Map(existing.map((trackState) => [trackState.track_id, trackState]));
  return tracks.map((track) => {
    const existingState = byId.get(track.trackId);
    if (!existingState) {
      return buildTrackRuntimeState(track);
    }
    return hydrateTrackRuntimeState(existingState, track);
  });
}

function getTrackStateOrThrow(status: CampaignStatus, trackId: string): TrackRuntimeState {
  const state = status.track_states.find((trackState) => trackState.track_id === trackId);
  if (!state) {
    throw new Error(`Missing runtime state for track ${trackId}.`);
  }
  return state;
}

function hasRunnableTracks(status: CampaignStatus, consecutiveLimit: number): boolean {
  if (deriveCampaignControlState({ status }).budgetState === "capped") {
    return false;
  }
  return status.track_states.some(
    (trackState) => canTrackStartNewEditTurn(trackState, {
      maxIterations: status.max_iterations,
      consecutiveLimit,
      maxEditTurnsPerTrack: MAX_EDIT_TURNS_PER_TRACK,
      noImprovementLimit: NO_IMPROVEMENT_LIMIT,
    }),
  );
}

export function canTrackStartNewEditTurn(
  trackState: Pick<TrackRuntimeState, "current_iteration" | "patience_streak" | "edit_turns_used">,
  options: {
    maxIterations: number;
    consecutiveLimit: number;
    maxEditTurnsPerTrack: number;
    noImprovementLimit: number;
  },
): boolean {
  if (trackState.current_iteration >= options.maxIterations) {
    return false;
  }
  if (trackState.patience_streak >= options.consecutiveLimit) {
    return false;
  }
  if (trackState.edit_turns_used >= options.maxEditTurnsPerTrack) {
    return false;
  }
  if (trackState.patience_streak >= options.noImprovementLimit) {
    return false;
  }
  return true;
}

export function deriveCampaignControlState({
  status,
  nowIso,
}: {
  status: Pick<CampaignStatus, "started_at" | "stage" | "stop_reason" | "leading_unverified_candidate" | "track_states"> & {
    campaign_id?: string;
  };
  nowIso?: string;
}): {
  campaignMode: CampaignMode;
  budgetState: BudgetState;
  stopReason: StopReason;
  elapsedMs: number;
} {
  const now = nowIso ? new Date(nowIso) : new Date();
  const startedAt = new Date(status.started_at || now.toISOString());
  const elapsedMs = Math.max(0, now.getTime() - startedAt.getTime());
  const budgetState: BudgetState = elapsedMs >= WALL_CLOCK_BUDGET_MS
    ? "capped"
    : elapsedMs >= WALL_CLOCK_BUDGET_MS * NEARING_CAP_RATIO
      ? "nearing_cap"
      : "healthy";
  const hasLeadingCandidate = Boolean(status.leading_unverified_candidate?.run_id);
  const hasNoImprovement = (status.track_states ?? []).some((trackState) => Number(trackState.patience_streak ?? 0) >= NO_IMPROVEMENT_LIMIT);
  const allTracksReachedFormal = (status.track_states ?? []).length > 0
    && (status.track_states ?? []).every((trackState) => Number(trackState.formal_runs_completed ?? 0) > 0);
  const campaignMode: CampaignMode = (budgetState === "capped" || hasLeadingCandidate || hasNoImprovement || allTracksReachedFormal)
    ? "closeout"
    : "exploration";
  const normalizedStopReason = normalizeStopReason(status.stop_reason);

  if (normalizedStopReason === "manual_stop_loss") {
    return {
      campaignMode: "closeout",
      budgetState,
      stopReason: "manual_stop_loss",
      elapsedMs,
    };
  }
  if (budgetState === "capped") {
    return {
      campaignMode,
      budgetState,
      stopReason: "wall_clock_cap",
      elapsedMs,
    };
  }
  return {
    campaignMode,
    budgetState,
    stopReason: hasNoImprovement && status.stage === "done" ? "no_improvement" : "none",
    elapsedMs,
  };
}

function refreshCampaignControlState(status: CampaignStatus): void {
  const control = deriveCampaignControlState({ status });
  status.campaign_mode = control.campaignMode;
  status.budget_state = control.budgetState;
  if (normalizeStopReason(status.stop_reason) !== "manual_stop_loss") {
    status.stop_reason = control.stopReason;
  }
}

function resolveTrackAllowedDirs(track: CampaignTrack): string[] {
  return track.allowedChangeScope.map((item) => resolveRepoPath(item));
}

function buildTrackRunId(campaignId: string, trackId: string, iteration: number): string {
  const sanitizedTrackId = trackId.replace(/[^a-zA-Z0-9_-]+/g, "_");
  return `${campaignId}-${sanitizedTrackId}-iter-${String(iteration).padStart(3, "0")}`;
}

function buildTrackOutputPath(rootDir: string, trackId: string, fileName: string): string {
  return path.join(rootDir, trackId, fileName);
}

function isCanonicalPromotionReady(track: CampaignTrack, formalMetrics: MetricSnapshot): boolean {
  return track.promotionTarget === "canonical_mainline" && formalMetrics.dataset_name === DEFAULT_CANONICAL_DATASET_NAME;
}

function primeCanonicalTrackLocalBest(status: CampaignStatus, tracks: CampaignTrack[]): void {
  const canonical = findCanonicalTrack(tracks);
  if (!canonical) {
    return;
  }
  const canonicalState = status.track_states.find((trackState) => trackState.track_id === canonical.trackId);
  if (!canonicalState) {
    return;
  }
  canonicalState.local_best = { ...status.accepted_stable_best };
}

function findCanonicalTrack(tracks: CampaignTrack[]): CampaignTrack | null {
  return tracks.find((track) => track.trackId === "canonical_mainline")
    ?? tracks.find((track) => track.promotionTarget === "canonical_mainline")
    ?? null;
}

function normalizeMetricSnapshot(value: unknown): MetricSnapshot | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Record<string, unknown>;
  return {
    source_path: String(record.source_path ?? ""),
    result_json: String(record.result_json ?? record.source_path ?? ""),
    dataset_name: String(record.dataset_name ?? ""),
    target_mode: String(record.target_mode ?? ""),
    target_space: String(record.target_space ?? ""),
    primary_metric_name: String(record.primary_metric_name ?? DEFAULT_PRIMARY_METRIC_NAME),
    val_primary_metric: toNumberOrNull(record.val_primary_metric),
    formal_val_primary_metric: toNumberOrNull(record.formal_val_primary_metric),
    val_rmse: toNumberOrNull(record.val_rmse),
    test_primary_metric: toNumberOrNull(record.test_primary_metric),
    test_rmse: toNumberOrNull(record.test_rmse),
    best_checkpoint_path: typeof record.best_checkpoint_path === "string" ? record.best_checkpoint_path : null,
    last_checkpoint_path: typeof record.last_checkpoint_path === "string" ? record.last_checkpoint_path : null,
    feature_family: typeof record.feature_family === "string" ? record.feature_family : null,
    model_family: typeof record.model_family === "string" ? record.model_family : null,
    experiment_track: typeof record.experiment_track === "string" ? record.experiment_track : null,
    evaluation_mode: typeof record.evaluation_mode === "string" ? record.evaluation_mode : null,
    artifacts: parseStringArray(record.artifacts),
  };
}

function syncAcceptedBestAlias(status: CampaignStatus) {
  status.accepted_best = { ...status.accepted_stable_best };
}

function normalizeStage(value: unknown): Stage {
  if (
    value === "baseline"
    || value === "bank_qc"
    || value === "bank_qc_failed"
    || value === "editing"
    || value === "smoke"
    || value === "formal_eval"
    || value === "accepted"
    || value === "rejected"
    || value === "rollback"
    || value === "done"
    || value === "paused"
  ) {
    return value;
  }
  return "baseline";
}

function normalizeCampaignMode(value: unknown): CampaignMode {
  return value === "closeout" ? "closeout" : "exploration";
}

function normalizeBudgetState(value: unknown): BudgetState {
  if (value === "nearing_cap" || value === "capped") {
    return value;
  }
  return "healthy";
}

function normalizeTrackOrigin(value: unknown): RuntimeTrackOrigin {
  return value === "incubation" ? "incubation" : "default";
}

function normalizeThreadItemUsageSummary(value: unknown): ThreadItemUsageSummary | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Record<string, unknown>;
  return {
    total_items: Number(record.total_items ?? 0),
    reasoning_items: Number(record.reasoning_items ?? 0),
    agent_messages: Number(record.agent_messages ?? 0),
    command_executions: Number(record.command_executions ?? 0),
    file_changes: Number(record.file_changes ?? 0),
    web_searches: Number(record.web_searches ?? 0),
    mcp_tool_calls: Number(record.mcp_tool_calls ?? 0),
    todo_lists: Number(record.todo_lists ?? 0),
    errors: Number(record.errors ?? 0),
    completed_items: Number(record.completed_items ?? 0),
    failed_items: Number(record.failed_items ?? 0),
  };
}

function normalizeStopReason(value: unknown): StopReason {
  if (value === "no_improvement" || value === "wall_clock_cap" || value === "manual_stop_loss") {
    return value;
  }
  return "none";
}

function normalizeAcceptedBest(value: unknown): AcceptedBestSnapshot {
  if (!value || typeof value !== "object") {
    return {
      run_id: "",
      dataset_name: "",
      target_mode: "",
      target_space: "",
      primary_metric_name: "",
      val_primary_metric: null,
      formal_val_primary_metric: null,
      val_rmse: null,
      test_primary_metric: null,
      test_rmse: null,
      feature_family: null,
      model_family: null,
      evaluation_mode: null,
      result_json: null,
      artifacts: [],
    };
  }

  const record = value as Record<string, unknown>;
  return {
    run_id: String(record.run_id ?? ""),
    dataset_name: String(record.dataset_name ?? ""),
    target_mode: String(record.target_mode ?? ""),
    target_space: String(record.target_space ?? ""),
    primary_metric_name: String(record.primary_metric_name ?? ""),
    val_primary_metric: toNumberOrNull(record.val_primary_metric),
    formal_val_primary_metric: toNumberOrNull(record.formal_val_primary_metric),
    val_rmse: toNumberOrNull(record.val_rmse),
    test_primary_metric: toNumberOrNull(record.test_primary_metric),
    test_rmse: toNumberOrNull(record.test_rmse),
    feature_family: typeof record.feature_family === "string" ? record.feature_family : null,
    model_family: typeof record.model_family === "string" ? record.model_family : null,
    evaluation_mode: typeof record.evaluation_mode === "string" ? record.evaluation_mode : null,
    result_json: typeof record.result_json === "string" ? record.result_json : null,
    artifacts: parseStringArray(record.artifacts),
  };
}

function normalizeCandidate(value: unknown): CandidateSnapshot {
  if (!value || typeof value !== "object") {
    return buildEmptyCandidate("campaign-candidate");
  }

  const record = value as Record<string, unknown>;
  return {
    run_id: String(record.run_id ?? "campaign-candidate"),
    stage: String(record.stage ?? "baseline"),
    track_id: String(record.track_id ?? ""),
    track_goal: String(record.track_goal ?? ""),
    promotion_target: String(record.promotion_target ?? ""),
    hypothesis: String(record.hypothesis ?? ""),
    why_this_change: String(record.why_this_change ?? ""),
    changes_summary: String(record.changes_summary ?? ""),
    change_bucket: String(record.change_bucket ?? ""),
    track_comparison_note: String(record.track_comparison_note ?? ""),
    files_touched: parseStringArray(record.files_touched),
    commands: parseStringArray(record.commands),
    search_queries: parseResearchQueries(record.search_queries),
    research_evidence: parseResearchEvidence(record.research_evidence),
    smoke_metrics: normalizeMetricSnapshot(record.smoke_metrics),
    final_metrics: normalizeMetricSnapshot(record.final_metrics),
    allowed_scope_ok: Boolean(record.allowed_scope_ok ?? false),
    rollback_applied: Boolean(record.rollback_applied ?? false),
    relevance_label: String(record.relevance_label ?? ""),
    relevance_reason: String(record.relevance_reason ?? ""),
    decision: String(record.decision ?? ""),
    next_step: String(record.next_step ?? ""),
    artifacts: parseStringArray(record.artifacts),
    tool_usage_summary: normalizeThreadItemUsageSummary(record.tool_usage_summary),
    thinking_heartbeat_at: String(record.thinking_heartbeat_at ?? ""),
    last_retrieval_at: String(record.last_retrieval_at ?? ""),
    last_decision_at: String(record.last_decision_at ?? ""),
    last_judgment_at: String(record.last_judgment_at ?? ""),
    last_materialization_at: String(record.last_materialization_at ?? ""),
    last_smoke_at: String(record.last_smoke_at ?? ""),
    stale_reason_codes: parseStringArray(record.stale_reason_codes),
    pivot_reason_codes: parseStringArray(record.pivot_reason_codes),
    search_budget_state: normalizeBudgetState(record.search_budget_state),
  };
}

function buildEmptyCandidate(runId: string): CandidateSnapshot {
  return {
    run_id: runId,
    stage: "baseline",
    track_id: "",
    track_goal: "",
    promotion_target: "",
    hypothesis: "",
    why_this_change: "",
    changes_summary: "",
    change_bucket: "",
    track_comparison_note: "",
    files_touched: [],
    commands: [],
    search_queries: [],
    research_evidence: [],
    smoke_metrics: null,
    final_metrics: null,
    allowed_scope_ok: false,
    rollback_applied: false,
    relevance_label: "",
    relevance_reason: "",
    decision: "pending",
    next_step: "",
    artifacts: [],
    tool_usage_summary: null,
    thinking_heartbeat_at: "",
    last_retrieval_at: "",
    last_decision_at: "",
    last_judgment_at: "",
    last_materialization_at: "",
    last_smoke_at: "",
    stale_reason_codes: [],
    pivot_reason_codes: [],
    search_budget_state: "healthy",
  };
}

function buildAcceptedBest({
  runId,
  primaryMetricName,
  smokeMetrics,
  formalMetrics,
}: {
  runId: string;
  primaryMetricName: string;
  smokeMetrics: MetricSnapshot;
  formalMetrics: MetricSnapshot | null;
}): AcceptedBestSnapshot {
  return {
    run_id: runId,
    dataset_name: formalMetrics?.dataset_name ?? smokeMetrics.dataset_name,
    target_mode: formalMetrics?.target_mode ?? smokeMetrics.target_mode,
    target_space: formalMetrics?.target_space ?? smokeMetrics.target_space,
    primary_metric_name: primaryMetricName,
    val_primary_metric: smokeMetrics.val_primary_metric,
    formal_val_primary_metric: formalMetrics?.val_primary_metric ?? null,
    val_rmse: formalMetrics?.val_rmse ?? smokeMetrics.val_rmse,
    test_primary_metric: formalMetrics?.test_primary_metric ?? null,
    test_rmse: formalMetrics?.test_rmse ?? null,
    feature_family: formalMetrics?.feature_family ?? smokeMetrics.feature_family ?? null,
    model_family: formalMetrics?.model_family ?? smokeMetrics.model_family ?? null,
    evaluation_mode: formalMetrics?.evaluation_mode ?? smokeMetrics.evaluation_mode ?? null,
    result_json: formalMetrics?.result_json ?? smokeMetrics.result_json ?? null,
    artifacts: dedupeStrings([
      smokeMetrics.source_path,
      smokeMetrics.best_checkpoint_path ?? undefined,
      smokeMetrics.last_checkpoint_path ?? undefined,
      formalMetrics?.source_path ?? undefined,
      formalMetrics?.best_checkpoint_path ?? undefined,
      formalMetrics?.last_checkpoint_path ?? undefined,
    ]),
  };
}

async function runCodexEditTurn({
  thread,
  campaignId,
  iteration,
  status,
  track,
  allowedDirs,
  programDocuments,
}: {
  thread: ReturnType<Codex["startThread"]> | ReturnType<Codex["resumeThread"]>;
  campaignId: string;
  iteration: number;
  status: CampaignStatus;
  track: CampaignTrack;
  allowedDirs: string[];
  programDocuments: Awaited<ReturnType<typeof loadProgramDocuments>>;
}): Promise<CodexEditResult> {
  const schema = {
    type: "object",
    properties: {
      hypothesis: { type: "string" },
      why_this_change: { type: "string" },
      changes_summary: { type: "string" },
      change_bucket: { type: "string" },
      track_comparison_note: { type: "string" },
      files_touched: { type: "array", items: { type: "string" } },
      next_step: { type: "string" },
      search_queries: {
        type: "array",
        items: {
          type: "object",
          properties: {
            search_query: { type: "string" },
            search_intent: { type: "string" },
          },
          required: ["search_query", "search_intent"],
          additionalProperties: false,
        },
      },
      research_evidence: {
        type: "array",
        items: {
          type: "object",
          properties: {
            search_query: { type: "string" },
            search_intent: { type: "string" },
            source_type: { type: "string" },
            source_url: { type: "string" },
            source_title: { type: "string" },
            why_it_matters: { type: "string" },
          },
          required: ["search_query", "search_intent", "source_type", "source_url", "source_title", "why_it_matters"],
          additionalProperties: false,
        },
      },
    },
    required: [
      "hypothesis",
      "why_this_change",
      "changes_summary",
      "change_bucket",
      "track_comparison_note",
      "files_touched",
      "next_step",
      "search_queries",
      "research_evidence",
    ],
    additionalProperties: false,
  } as const;

  const prompt = [
    buildCodexPrompt({
      campaignId,
      iteration,
      allowedDirs,
      programDocuments,
      track,
      statusSnapshot: {
        stage: status.stage,
        acceptedStableBestMetric: status.accepted_stable_best.val_primary_metric,
        acceptedStableBestSource: status.accepted_stable_best.artifacts[0] ?? "",
        canonicalPromotionMetric: status.accepted_stable_best.primary_metric_name || DEFAULT_PRIMARY_METRIC_NAME,
      },
      promptMode: status.campaign_mode === "closeout" ? "closeout" : "exploration",
    }),
    "",
    "当前状态：",
    summarizeStatusForPrompt(status, track.trackId),
  ].join("\n");

  const turn = await thread.run(prompt, { outputSchema: schema });
  const proposal = parseProposal(turn.finalResponse);
  return {
    proposal,
    items: turn.items,
    threadId: thread.id ?? null,
  };
}

function summarizeStatusForPrompt(status: CampaignStatus, activeTrackId: string): string {
  const lines = [
    `- stage: ${status.stage}`,
    `- campaign_id: ${status.campaign_id}`,
    `- campaign_mode: ${status.campaign_mode}`,
    `- budget_state: ${status.budget_state}`,
    `- stop_reason: ${status.stop_reason}`,
    `- active_track_id: ${status.active_track_id || activeTrackId}`,
    `- current_iteration: ${status.current_iteration}/${status.max_iterations}`,
    `- patience: ${status.patience}`,
    `- current_command: ${status.current_command || "none"}`,
  ];

  if (status.accepted_stable_best.run_id) {
    lines.push(
      `- accepted_stable_best: ${status.accepted_stable_best.run_id} (${status.accepted_stable_best.primary_metric_name}=${status.accepted_stable_best.formal_val_primary_metric ?? status.accepted_stable_best.val_primary_metric ?? "null"})`,
    );
  }

  if (status.candidate?.run_id) {
    lines.push(
      `- current_candidate: ${status.candidate.run_id} stage=${status.candidate.stage} decision=${status.candidate.decision}`,
    );
    if (status.candidate.change_bucket) {
      lines.push(`- current_candidate_bucket: ${status.candidate.change_bucket}`);
    }
    if (status.candidate.relevance_label) {
      lines.push(`- current_candidate_relevance: ${status.candidate.relevance_label}`);
    }
  }

  const activeTracks = status.track_states
    .map((trackState) => ({
      trackId: trackState.track_id,
      iteration: trackState.current_iteration,
      patience: trackState.patience_streak,
      localBest: trackState.local_best.formal_val_primary_metric ?? trackState.local_best.val_primary_metric,
      lastDecision: trackState.last_decision,
    }))
    .sort((left, right) => {
      if (left.trackId === activeTrackId) return -1;
      if (right.trackId === activeTrackId) return 1;
      return left.trackId.localeCompare(right.trackId);
    })
    .slice(0, 4);

  if (activeTracks.length > 0) {
    lines.push("- track_state_summary:");
    for (const trackState of activeTracks) {
      lines.push(
        `  - ${trackState.trackId}: iter=${trackState.iteration}, patience=${trackState.patience}, local_best=${trackState.localBest ?? "null"}, last_decision=${trackState.lastDecision || "none"}`,
      );
    }
  }

  return lines.join("\n");
}

export function parseProposal(text: string): {
  hypothesis: string;
  why_this_change: string;
  changes_summary: string;
  change_bucket: string;
  track_comparison_note: string;
  files_touched: string[];
  next_step: string;
  search_queries: ResearchQuery[];
  research_evidence: ResearchEvidence[];
} {
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    return {
      hypothesis: String(parsed.hypothesis ?? ""),
      why_this_change: String(parsed.why_this_change ?? ""),
      changes_summary: String(parsed.changes_summary ?? ""),
      change_bucket: String(parsed.change_bucket ?? ""),
      track_comparison_note: String(parsed.track_comparison_note ?? ""),
      files_touched: parseStringArray(parsed.files_touched),
      next_step: String(parsed.next_step ?? ""),
      search_queries: parseResearchQueries(parsed.search_queries),
      research_evidence: parseResearchEvidence(parsed.research_evidence),
    };
  } catch {
    return {
      hypothesis: text.trim(),
      why_this_change: "",
      changes_summary: "",
      change_bucket: "",
      track_comparison_note: "",
      files_touched: [],
      next_step: "",
      search_queries: [],
      research_evidence: [],
    };
  }
}

function auditFileChanges(items: ThreadItem[], allowedDirs: string[]): {
  allowedScopeOk: boolean;
  filesTouched: string[];
  fileChanges: Array<{ path: string; kind: string }>;
  violations: string[];
} {
  const fileChanges = items.filter((item): item is FileChangeItem => item.type === "file_change");
  const seen = new Set<string>();
  const touched: string[] = [];
  const violations: string[] = [];
  const normalizedAllowed = allowedDirs.map((dir) => normalizeRepoPath(dir));

  for (const item of fileChanges) {
    for (const change of item.changes) {
      const rel = normalizeRepoPath(change.path);
      if (!seen.has(rel)) {
        seen.add(rel);
        touched.push(rel);
      }

      if (!isAllowedPath(rel, normalizedAllowed)) {
        violations.push(rel);
      }
    }
    if (item.status !== "completed") {
      violations.push(`file_change:${item.id}:failed`);
    }
  }

  return {
    allowedScopeOk: violations.length === 0,
    filesTouched: touched,
    fileChanges: fileChanges.flatMap((item) => item.changes.map((change) => ({ path: normalizeRepoPath(change.path), kind: change.kind }))),
    violations,
  };
}

export function classifyCommandRelevance(
  filesTouched: string[],
  smokeCommand: string,
  formalCommand: string,
): RelevanceClassification {
  const relevantPatterns = deriveRelevantPatterns(smokeCommand, formalCommand);
  if (filesTouched.length === 0) {
    return {
      label: "off_track_but_ran",
      reason: "这轮没有真正改到任何文件，所以即使继续跑也不会产生新的实验内容。",
      hardBlock: false,
      relevantPatterns,
      coreFiles: [],
      supportingFiles: [],
      offTrackFiles: [],
    };
  }

  const coreFiles = filesTouched.filter((item) => relevantPatterns.some((pattern) => new RegExp(pattern).test(item)));
  const supportingFiles = filesTouched.filter((item) => SUPPORTING_ALLOWLIST.some((pattern) => pattern.test(item)));
  const offTrackFiles = filesTouched.filter((item) => !coreFiles.includes(item) && !supportingFiles.includes(item));

  if (coreFiles.length > 0) {
    return {
      label: "on_track",
      reason: supportingFiles.length > 0
        ? "这轮既包含命中当前实验轨的核心改动，也顺手补了配套测试或接线文件，所以先放行再看结果。"
        : "这轮直接改到了当前实验轨真正会执行的核心文件，属于正面命中当前方法路线。",
      hardBlock: false,
      relevantPatterns,
      coreFiles,
      supportingFiles,
      offTrackFiles,
    };
  }

  if (supportingFiles.length > 0 && offTrackFiles.length === 0) {
    return {
      label: "supporting_change",
      reason: "这轮主要在补测试、结果落盘或命令行接线，它不一定直接改变模型结构，但确实影响这条轨道能不能稳定出分。",
      hardBlock: false,
      relevantPatterns,
      coreFiles,
      supportingFiles,
      offTrackFiles,
    };
  }

  return {
    label: "exploratory_but_indirect",
    reason: "这轮没有直接命中当前方法的核心脚本，但它仍然落在允许目录里，所以系统会先让它跑，再根据结果判断是否值得保留。",
    hardBlock: false,
    relevantPatterns,
    coreFiles,
    supportingFiles,
    offTrackFiles,
  };
}

function deriveRelevantPatterns(smokeCommand: string, formalCommand: string): string[] {
  const combined = `${smokeCommand}\n${formalCommand}`;
  const patterns = new Set<string>();

  if (combined.includes("train_ridge.py")) {
    patterns.add("^scripts/train_ridge\\.py$");
    patterns.add("^src/bci_autoresearch/features/.+$");
  }
  if (combined.includes("train_feature_lstm.py")) {
    patterns.add("^scripts/train_feature_lstm\\.py$");
    patterns.add("^src/bci_autoresearch/features/.+$");
    patterns.add("^src/bci_autoresearch/models/.+$");
  }
  if (combined.includes("train_feature_gru.py") || combined.includes("feature_gru")) {
    patterns.add("^scripts/train_feature_gru\\.py$");
    patterns.add("^src/bci_autoresearch/features/.+$");
    patterns.add("^src/bci_autoresearch/models/.+$");
  }
  if (combined.includes("train_feature_tcn.py") || combined.includes("feature_tcn")) {
    patterns.add("^scripts/train_feature_tcn\\.py$");
    patterns.add("^src/bci_autoresearch/features/.+$");
    patterns.add("^src/bci_autoresearch/models/.+$");
  }
  if (combined.includes("train_lstm.py")) {
    patterns.add("^scripts/train_lstm\\.py$");
    patterns.add("^src/bci_autoresearch/models/.+$");
  }
  if (patterns.size === 0) {
    patterns.add("^scripts/train_.+\\.py$");
    patterns.add("^src/bci_autoresearch/features/.+$");
    patterns.add("^src/bci_autoresearch/models/.+$");
  }
  return Array.from(patterns);
}

function collectCommandStrings(items: ThreadItem[]): string[] {
  const commands: string[] = [];
  for (const item of items) {
    if (item.type === "command_execution") {
      commands.push(item.command);
    }
  }
  return dedupeStrings(commands);
}

function isAllowedPath(relPath: string, allowedDirs: string[]): boolean {
  const normalized = relPath.split(path.sep).join("/");
  if (DENYLIST_PATHS.some((pattern) => pattern.test(normalized))) {
    return false;
  }
  if (ALLOWLIST.some((pattern) => pattern.test(normalized))) {
    return true;
  }
  if (SUPPORTING_ALLOWLIST.some((pattern) => pattern.test(normalized))) {
    return true;
  }
  return allowedDirs.some((dir) => {
    const normalizedDir = normalizeRepoPath(dir);
    return normalized === normalizedDir || normalized.startsWith(`${normalizedDir}/`);
  });
}

function parseResearchQueries(value: unknown): ResearchQuery[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => normalizeResearchQuery(item))
    .filter((item): item is ResearchQuery => item !== null);
}

function normalizeResearchQuery(value: unknown): ResearchQuery | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const searchQuery = typeof record.search_query === "string" ? record.search_query.trim() : "";
  const searchIntent = typeof record.search_intent === "string" ? record.search_intent.trim() : "";
  if (!searchQuery) {
    return null;
  }
  return {
    search_query: searchQuery,
    search_intent: searchIntent || "general",
  };
}

function parseResearchEvidence(value: unknown): ResearchEvidence[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => normalizeResearchEvidence(item))
    .filter((item): item is ResearchEvidence => item !== null);
}

function normalizeResearchEvidence(value: unknown): ResearchEvidence | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const base = normalizeResearchQuery(record);
  const sourceType = typeof record.source_type === "string" ? record.source_type.trim() : "";
  const sourceUrl = typeof record.source_url === "string" ? record.source_url.trim() : "";
  const sourceTitle = typeof record.source_title === "string" ? record.source_title.trim() : "";
  const whyItMatters = typeof record.why_it_matters === "string" ? record.why_it_matters.trim() : "";
  if (!base || !sourceType || !sourceUrl || !sourceTitle || !whyItMatters) {
    return null;
  }
  return {
    ...base,
    source_type: sourceType,
    source_url: sourceUrl,
    source_title: sourceTitle,
    why_it_matters: whyItMatters,
  };
}

export function buildResearchLogEntries({
  campaignId,
  runId,
  trackId,
  recordedAt,
  searchQueries,
  researchEvidence,
}: {
  campaignId: string;
  runId: string;
  trackId: string;
  recordedAt: string;
  searchQueries: ResearchQuery[];
  researchEvidence: ResearchEvidence[];
}): {
  queryEntries: ResearchQueryLogEntry[];
  evidenceEntries: ResearchEvidenceLogEntry[];
} {
  return {
    queryEntries: searchQueries.map((query) => ({
      campaign_id: campaignId,
      run_id: runId,
      track_id: trackId,
      recorded_at: recordedAt,
      used_in_run_id: runId,
      ...query,
    })),
    evidenceEntries: researchEvidence.map((evidence) => ({
      campaign_id: campaignId,
      run_id: runId,
      track_id: trackId,
      recorded_at: recordedAt,
      used_in_run_id: runId,
      ...evidence,
    })),
  };
}

async function appendJsonl(filePath: string, rows: Array<object>): Promise<void> {
  if (rows.length === 0) {
    return;
  }
  const payload = rows.map((row) => `${JSON.stringify(row)}\n`).join("");
  await writeFile(filePath, payload, { flag: "a" });
}

async function snapshotPaths(roots: string[]): Promise<Map<string, SnapshotEntry>> {
  const snapshot = new Map<string, SnapshotEntry>();
  for (const root of roots) {
    const abs = resolveRepoPath(root);
    await snapshotTree(abs, snapshot);
  }
  return snapshot;
}

async function snapshotTree(currentPath: string, snapshot: Map<string, SnapshotEntry>): Promise<void> {
  const entries = await readdir(currentPath, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name === ".git" || entry.name === "node_modules" || entry.name === ".venv" || entry.name === "__pycache__" || entry.name === "dist") {
      continue;
    }
    const abs = path.join(currentPath, entry.name);
    if (entry.isDirectory()) {
      await snapshotTree(abs, snapshot);
      continue;
    }
    if (!entry.isFile()) {
      continue;
    }
    const content = await readFile(abs, "utf8");
    snapshot.set(abs, { existed: true, content });
  }
}

async function restoreSnapshot(snapshot: Map<string, SnapshotEntry>, touchedFiles: string[] = []) {
  const touched = new Set(touchedFiles.map((item) => resolveRepoPath(item)));
  for (const [absPath, entry] of snapshot.entries()) {
    if (!entry.existed) {
      await rm(absPath, { force: true });
      continue;
    }
    await mkdir(path.dirname(absPath), { recursive: true });
    await writeFile(absPath, entry.content ?? "", "utf8");
  }

  for (const absPath of touched) {
    if (snapshot.has(absPath)) {
      continue;
    }
    await rm(absPath, { force: true });
  }
}

async function runShellCommand(
  command: string,
  cwd: string,
  metadata?: {
    campaignId?: string;
    trackId?: string | null;
    taskKind?: string;
    modelFamily?: string | null;
    priority?: number;
    expectedMemoryClass?: "high" | "low" | "service";
  },
): Promise<RunCommandResult> {
  const startedAt = Date.now();
  const shell = process.platform === "win32" ? "cmd.exe" : process.env.SHELL ?? "/bin/zsh";
  const args = process.platform === "win32" ? ["/c", command] : ["-lc", command];

  return await new Promise<RunCommandResult>((resolve, reject) => {
    const child = spawn(shell, args, {
      cwd,
      env: sanitizeLaunchEnvironment(process.env),
      stdio: ["ignore", "pipe", "pipe"],
    });

    const registerPromise = registerManagedProcess(PROCESS_REGISTRY_PATH, {
      pid: child.pid ?? -1,
      campaignId: metadata?.campaignId ?? "standalone",
      trackId: metadata?.trackId ?? null,
      taskKind: metadata?.taskKind ?? "shell_command",
      modelFamily: metadata?.modelFamily ?? null,
      priority: metadata?.priority ?? 3,
      expectedMemoryClass: metadata?.expectedMemoryClass ?? "low",
      command,
      startedAt: new Date(startedAt).toISOString(),
      cwd,
    }).catch(() => undefined);

    let stdout = "";
    let stderr = "";

    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });

    child.on("error", (error) => {
      void unregisterManagedProcess(PROCESS_REGISTRY_PATH, child.pid ?? -1);
      reject(error);
    });

    child.on("close", (exitCode) => {
      void Promise.allSettled([
        registerPromise,
        unregisterManagedProcess(PROCESS_REGISTRY_PATH, child.pid ?? -1),
      ]).finally(() => {
        resolve({
          command,
          exitCode: exitCode ?? -1,
          stdout,
          stderr,
          durationMs: Date.now() - startedAt,
        });
      });
    });
  });
}

function ensureOutputJson(command: string, outputPath: string): string {
  if (/\B--output-json(?:=|\s)/.test(command) || /\B--output_json(?:=|\s)/.test(command)) {
    return command;
  }
  return `${command} --output-json ${shellQuote(outputPath)}`;
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

async function recordIteration({
  status,
  iteration,
  runId,
  parentRunId,
  track,
  smokeMetrics,
  finalMetrics,
  filesTouched,
  commands,
  searchQueries,
  researchEvidence,
  allowedScopeOk,
  rollbackApplied,
  relevanceLabel,
  relevanceReason,
  decision,
  nextStep,
  hypothesis,
  whyThisChange,
  changesSummary,
  changeBucket,
  trackComparisonNote,
  artifacts,
  telemetry,
  reviewPacketDir,
}: {
  status: CampaignStatus;
  iteration: number;
  runId: string;
  parentRunId: string | null;
  track: CampaignTrack | null;
  smokeMetrics: MetricSnapshot | null;
  finalMetrics: MetricSnapshot | null;
  filesTouched: string[];
  commands: string[];
  searchQueries: ResearchQuery[];
  researchEvidence: ResearchEvidence[];
  allowedScopeOk: boolean;
  rollbackApplied: boolean;
  relevanceLabel: string;
  relevanceReason: string;
  decision: string;
  nextStep: string;
  hypothesis: string;
  whyThisChange: string;
  changesSummary: string;
  changeBucket: string;
  trackComparisonNote: string;
  artifacts: string[];
  telemetry: TurnTelemetryState;
  reviewPacketDir: string;
}) {
  const record: CampaignRecord = {
    campaign_id: status.campaign_id,
    run_id: runId,
    parent_run_id: parentRunId,
    iteration,
    stage: status.stage,
    recorded_at: new Date().toISOString(),
    agent_name: DEFAULT_AGENT_NAME,
    track_id: track?.trackId ?? "",
    track_goal: track?.trackGoal ?? "",
    promotion_target: track?.promotionTarget ?? "",
    dataset_name: finalMetrics?.dataset_name ?? smokeMetrics?.dataset_name ?? status.accepted_stable_best.dataset_name,
    target_mode: finalMetrics?.target_mode ?? smokeMetrics?.target_mode ?? status.accepted_stable_best.target_mode,
    target_space: finalMetrics?.target_space ?? smokeMetrics?.target_space ?? status.accepted_stable_best.target_space,
    primary_metric_name: status.accepted_stable_best.primary_metric_name || DEFAULT_PRIMARY_METRIC_NAME,
    hypothesis,
    why_this_change: whyThisChange,
    changes_summary: changesSummary,
    change_bucket: changeBucket,
    track_comparison_note: trackComparisonNote,
    files_touched: filesTouched,
    commands,
    search_queries: searchQueries,
    research_evidence: researchEvidence,
    smoke_metrics: smokeMetrics,
    final_metrics: finalMetrics,
    allowed_scope_ok: allowedScopeOk,
    rollback_applied: rollbackApplied,
    relevance_label: relevanceLabel,
    relevance_reason: relevanceReason,
    decision,
    next_step: nextStep,
    artifacts,
    tool_usage_summary: telemetry.tool_usage_summary,
    thinking_heartbeat_at: telemetry.thinking_heartbeat_at,
    last_retrieval_at: telemetry.last_retrieval_at,
    last_decision_at: telemetry.last_decision_at,
    last_judgment_at: telemetry.last_judgment_at,
    last_materialization_at: telemetry.last_materialization_at,
    last_smoke_at: telemetry.last_smoke_at,
    stale_reason_codes: telemetry.stale_reason_codes,
    pivot_reason_codes: telemetry.pivot_reason_codes,
    search_budget_state: telemetry.search_budget_state,
  };

  const line = `${JSON.stringify(record)}\n`;
  await writeFile(TOOLS_LEDGER_PATH, line, { flag: "a" });
  await writeFile(MONITOR_LEDGER_PATH, line, { flag: "a" });
  const researchEntries = buildResearchLogEntries({
    campaignId: status.campaign_id,
    runId,
    trackId: track?.trackId ?? "",
    recordedAt: record.recorded_at,
    searchQueries,
    researchEvidence,
  });
  await appendJsonl(RESEARCH_QUERIES_PATH, researchEntries.queryEntries);
  await appendJsonl(RESEARCH_EVIDENCE_PATH, researchEntries.evidenceEntries);
  await writeDailyReviewPacket(status.campaign_id, reviewPacketDir);
}

function summarizeMetrics(metrics: Record<string, unknown> | null, sourcePath: string): MetricSnapshot | null {
  if (!metrics) {
    return null;
  }

  const datasetName = String(metrics.dataset_name ?? "");
  const targetMode = String(metrics.target_mode ?? "");
  const targetSpace = String(metrics.target_space ?? "");
  const primaryMetricName = String(metrics.primary_metric ?? DEFAULT_PRIMARY_METRIC_NAME);
  const valPrimary = extractNumber(metrics, "val_metrics.mean_pearson_r_zero_lag_macro", "mean_pearson_r_zero_lag_macro", "val_r_zero", "val_r");
  const valRmse = extractNumber(metrics, "val_metrics.mean_rmse_deg_macro", "val_metrics.mean_rmse_macro", "val_rmse");
  const testPrimary = extractNumber(metrics, "test_metrics.mean_pearson_r_zero_lag_macro", "test_r_zero", "test_r");
  const bestCheckpoint = String(metrics.best_checkpoint_path ?? "");
  const lastCheckpoint = String(metrics.last_checkpoint_path ?? "");
  const trainSummary = getValueByPath(metrics, "train_summary");
  const featureFamilies = Array.isArray((trainSummary as Record<string, unknown> | undefined)?.feature_families)
    ? ((trainSummary as Record<string, unknown>).feature_families as unknown[])
        .filter((item): item is string => typeof item === "string" && item.trim() !== "")
    : [];
  const featureFamily = featureFamilies.length > 0 ? featureFamilies.join("+") : null;
  const modelFamily = typeof (trainSummary as Record<string, unknown> | undefined)?.model_family === "string"
    ? String((trainSummary as Record<string, unknown>).model_family)
    : null;
  const experimentTrack = typeof metrics.experiment_track === "string" ? metrics.experiment_track : null;
  const evaluationMode = experimentTrack === "within_session_upper_bound" ? "upper_bound_same_session" : "cross_session_mainline";

  return {
    source_path: sourcePath,
    result_json: sourcePath,
    dataset_name: datasetName,
    target_mode: targetMode,
    target_space: targetSpace,
    primary_metric_name: primaryMetricName,
    val_primary_metric: valPrimary,
    formal_val_primary_metric: valPrimary,
    val_rmse: valRmse,
    test_primary_metric: testPrimary,
    test_rmse: extractNumber(metrics, "test_metrics.mean_rmse_deg_macro", "test_metrics.mean_rmse_macro", "test_rmse"),
    best_checkpoint_path: bestCheckpoint || null,
    last_checkpoint_path: lastCheckpoint || null,
    feature_family: featureFamily,
    model_family: modelFamily,
    experiment_track: experimentTrack,
    evaluation_mode: evaluationMode,
    artifacts: dedupeStrings([sourcePath, bestCheckpoint || undefined, lastCheckpoint || undefined]),
  };
}

function extractPrimaryMetric(metrics: Record<string, unknown>): number | null {
  return extractNumber(metrics, "val_metrics.mean_pearson_r_zero_lag_macro", "mean_pearson_r_zero_lag_macro", "val_r_zero", "val_r");
}

function extractNumber(record: Record<string, unknown>, ...paths: string[]): number | null {
  for (const pathSpec of paths) {
    const value = getValueByPath(record, pathSpec);
    const numeric = toNumberOrNull(value);
    if (numeric !== null) {
      return numeric;
    }
  }
  return null;
}

function getValueByPath(record: Record<string, unknown>, pathSpec: string): unknown {
  const parts = pathSpec.split(".");
  let current: unknown = record;
  for (const part of parts) {
    if (!current || typeof current !== "object") {
      return undefined;
    }
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function toNumberOrNull(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return numeric;
    }
  }
  return null;
}

function isFiniteNumber(value: number | null): boolean {
  return value !== null && Number.isFinite(value);
}

function isBetterThan(candidate: number | null, current: number | null): boolean {
  if (candidate === null) {
    return false;
  }
  if (current === null) {
    return true;
  }
  return candidate > current + MIN_IMPROVEMENT_EPSILON;
}

function parseAllowedDirs(value: unknown): string[] {
  const dirs = parseStringArray(value);
  if (dirs.length > 0) {
    return dirs.map((dir) => resolveRepoPath(dir));
  }
  return DEFAULT_ALLOWED_DIRS;
}

function parseStringArray(value: unknown): string[] {
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string" && item.trim() !== "");
  }
  return [];
}

function parsePositiveInt(value: unknown, fallback: number): number {
  const numeric = typeof value === "string" ? Number.parseInt(value, 10) : typeof value === "number" ? Math.trunc(value) : NaN;
  if (Number.isFinite(numeric) && numeric > 0) {
    return numeric;
  }
  return fallback;
}

function parseBoolean(value: unknown): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    return value === "1" || value.toLowerCase() === "true" || value.toLowerCase() === "yes";
  }
  return false;
}

function optionalString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

function parseArgs(argv: string[]): ParsedArgs {
  const result: ParsedArgs = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }

    const inline = token.slice(2).split("=", 2);
    const key = inline[0];
    if (inline.length === 2) {
      result[key] = inline[1];
      continue;
    }

    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      result[key] = next;
      index += 1;
    } else {
      result[key] = true;
    }
  }
  return result;
}

async function readJsonIfExists<T>(filePath: string): Promise<T | null> {
  try {
    const text = await readFile(filePath, "utf8");
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

async function writeDailyReviewPacket(campaignId: string, reviewPacketDir: string): Promise<void> {
  const records = await readCampaignRecords(TOOLS_LEDGER_PATH);
  const today = new Date().toISOString().slice(0, 10);
  const packet = buildDailyReviewPacket({
    campaignId,
    reviewDate: today,
    records,
  });
  const packetDir = path.join(reviewPacketDir, today);
  await mkdir(packetDir, { recursive: true });
  await writeFile(path.join(packetDir, `${campaignId}.json`), `${JSON.stringify(packet, null, 2)}\n`, "utf8");
  await writeFile(path.join(packetDir, `${campaignId}.md`), `${renderDailyReviewPacketMarkdown(packet)}\n`, "utf8");
}

async function readCampaignRecords(filePath: string): Promise<CampaignRecord[]> {
  try {
    const text = await readFile(filePath, "utf8");
    return text
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line) as CampaignRecord);
  } catch {
    return [];
  }
}

async function writeStatus(status: CampaignStatus) {
  status.current_iteration = status.track_states.reduce((sum, trackState) => sum + trackState.current_iteration, 0);
  status.patience_streak = status.track_states.reduce(
    (maxValue, trackState) => Math.max(maxValue, trackState.patience_streak),
    0,
  );
  refreshCampaignControlState(status);
  syncAcceptedBestAlias(status);
  const tmpPath = `${STATUS_PATH}.${process.pid}.tmp`;
  await mkdir(path.dirname(STATUS_PATH), { recursive: true });
  await writeFile(tmpPath, `${JSON.stringify(status, null, 2)}\n`, "utf8");
  await rename(tmpPath, STATUS_PATH);
}

function resolveRepoPath(value: string): string {
  return path.isAbsolute(value) ? value : path.resolve(REPO_ROOT, value);
}

function normalizeRepoPath(value: string): string {
  return path.relative(REPO_ROOT, resolveRepoPath(value)).split(path.sep).join("/");
}

function dedupeStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    if (!value) {
      continue;
    }
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }
  return result;
}

function stringifyError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function isEntrypoint(): boolean {
  const argv1 = process.argv[1];
  if (!argv1) {
    return false;
  }
  return pathToFileURL(path.resolve(argv1)).href === import.meta.url;
}

if (isEntrypoint()) {
  main().catch((error) => {
    process.stderr.write(`${stringifyError(error)}\n`);
    process.exitCode = 1;
  });
}
