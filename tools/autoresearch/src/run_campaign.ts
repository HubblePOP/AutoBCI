import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Codex, type FileChangeItem, type ThreadItem } from "@openai/codex-sdk";

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

interface MetricSnapshot {
  source_path: string;
  result_json: string;
  dataset_name: string;
  target_mode: string;
  target_space: string;
  primary_metric_name: string;
  val_primary_metric: number | null;
  formal_val_primary_metric: number | null;
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
  hypothesis: string;
  why_this_change: string;
  changes_summary: string;
  files_touched: string[];
  commands: string[];
  smoke_metrics: MetricSnapshot | null;
  final_metrics: MetricSnapshot | null;
  allowed_scope_ok: boolean;
  rollback_applied: boolean;
  decision: string;
  next_step: string;
  artifacts: string[];
}

interface CampaignStatus {
  campaign_id: string;
  current_iteration: number;
  max_iterations: number;
  patience: number;
  stage: Stage;
  frozen_baseline: AcceptedBestSnapshot;
  accepted_stable_best: AcceptedBestSnapshot;
  leading_unverified_candidate: AcceptedBestSnapshot;
  accepted_best: AcceptedBestSnapshot;
  candidate: CandidateSnapshot;
  current_command: string;
  updated_at: string;
  codex_thread_id?: string | null;
  patience_streak?: number;
  last_error?: string | null;
}

interface CampaignRecord {
  campaign_id: string;
  run_id: string;
  parent_run_id: string | null;
  iteration: number;
  stage: string;
  recorded_at: string;
  agent_name: string;
  dataset_name: string;
  target_mode: string;
  target_space: string;
  primary_metric_name: string;
  hypothesis: string;
  why_this_change: string;
  changes_summary: string;
  files_touched: string[];
  commands: string[];
  smoke_metrics: MetricSnapshot | null;
  final_metrics: MetricSnapshot | null;
  allowed_scope_ok: boolean;
  rollback_applied: boolean;
  decision: string;
  next_step: string;
  artifacts: string[];
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
    files_touched: string[];
    next_step: string;
  };
  items: ThreadItem[];
  threadId: string | null;
}

interface RelevanceAudit {
  relevantScopeOk: boolean;
  relevantPatterns: string[];
  violations: string[];
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TOOLS_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(TOOLS_ROOT, "..", "..");

const STATUS_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "autoresearch_status.json");
const TOOLS_LEDGER_PATH = path.join(TOOLS_ROOT, "experiment_ledger.jsonl");
const MONITOR_LEDGER_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "experiment_ledger.jsonl");
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
const MIN_IMPROVEMENT_EPSILON = 1e-9;

const ALLOWLIST = [
  /^scripts\/train_[^/]+\.py$/,
  /^scripts\/build_monitor_artifacts\.py$/,
  /^scripts\/analyze_channel_halves\.py$/,
  /^src\/bci_autoresearch\/models\/.+$/,
  /^src\/bci_autoresearch\/features\/.+$/,
];

const DENYLIST_PATHS = [
  /^scripts\/convert_session\.py$/,
  /^src\/bci_autoresearch\/data\/.+$/,
  /^data\/.+$/,
  /^\/Volumes\/.+$/,
];

async function main() {
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
  const smokeOutputDir = resolveRepoPath(String(args["smoke-output-dir"] ?? args["smoke_output_dir"] ?? DEFAULT_SMOKE_OUTPUT_DIR));
  const formalOutputDir = resolveRepoPath(String(args["formal-output-dir"] ?? args["formal_output_dir"] ?? DEFAULT_SMOKE_OUTPUT_DIR));
  const dryRun = parseBoolean(args["dry-run"] ?? args["dry_run"]);

  await mkdir(path.dirname(STATUS_PATH), { recursive: true });
  await mkdir(TOOLS_ROOT, { recursive: true });
  await mkdir(path.dirname(TOOLS_LEDGER_PATH), { recursive: true });
  await mkdir(path.dirname(MONITOR_LEDGER_PATH), { recursive: true });
  await mkdir(smokeOutputDir, { recursive: true });
  await mkdir(formalOutputDir, { recursive: true });

  const existingStatus = await readJsonIfExists<Partial<CampaignStatus>>(STATUS_PATH);
  const status = buildInitialStatus({
    campaignId,
    maxIterations,
    patience,
    existingStatus,
    baselineMetricsPath,
    allowedDirs,
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
    bankQcCommand,
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

  const codex = new Codex();
  const thread = status.codex_thread_id
    ? codex.resumeThread(status.codex_thread_id, {
        workingDirectory: REPO_ROOT,
        skipGitRepoCheck: true,
        sandboxMode: "workspace-write",
        approvalPolicy: "never",
        networkAccessEnabled: false,
        additionalDirectories: allowedDirs,
      })
    : codex.startThread({
        workingDirectory: REPO_ROOT,
        skipGitRepoCheck: true,
        sandboxMode: "workspace-write",
        approvalPolicy: "never",
        networkAccessEnabled: false,
        additionalDirectories: allowedDirs,
      });

  const consecutiveLimit = patience > 0 ? patience : Number.POSITIVE_INFINITY;
  let consecutiveFailures = status.patience_streak ?? 0;

  while (status.current_iteration < status.max_iterations && consecutiveFailures < consecutiveLimit) {
    const parentRunId = status.accepted_stable_best.run_id || null;
    const iteration = status.current_iteration + 1;
    const runId = `${campaignId}-iter-${String(iteration).padStart(3, "0")}`;
    const smokeOutputPath = path.join(smokeOutputDir, `${runId}_smoke.json`);
    const formalOutputPath = path.join(formalOutputDir, `${runId}_formal.json`);

    const iterationBankQcReady = await runBankQcGate({
      status,
      campaignId,
      runId: `${runId}-bank-qc`,
      iteration,
      parentRunId,
      bankQcCommand,
      dryRun,
    });
    if (!iterationBankQcReady) {
      break;
    }

    const snapshot = await snapshotPaths(allowedDirs);
    status.stage = "editing";
    status.current_iteration = iteration;
    status.current_command = "codex:edit";
    status.last_error = null;
    status.candidate = {
      ...buildEmptyCandidate(runId),
      stage: "editing",
      decision: "editing",
      next_step: "等待候选改动生成。",
    };
    status.updated_at = new Date().toISOString();
    await writeStatus(status);

    let codexResult: CodexEditResult;
    let touchedFiles: string[] = [];
    try {
      codexResult = await runCodexEditTurn({
        thread,
        campaignId,
        iteration,
        status,
        allowedDirs,
        smokeCommand,
        formalCommand,
      });
      status.codex_thread_id = codexResult.threadId;
    } catch (error) {
      await restoreSnapshot(snapshot, touchedFiles);
      status.stage = "paused";
      status.current_command = "";
      status.last_error = stringifyError(error);
      status.candidate = buildEmptyCandidate(runId);
      status.candidate.rollback_applied = true;
      status.candidate.allowed_scope_ok = false;
      status.candidate.decision = "codex_failed";
      status.candidate.next_step = "修复代理提示或命令，再继续下一轮。";
      status.candidate.artifacts = [STATUS_PATH, TOOLS_LEDGER_PATH, MONITOR_LEDGER_PATH];
      status.updated_at = new Date().toISOString();
      await recordIteration({
        status,
        iteration,
        runId,
        parentRunId,
        smokeMetrics: null,
        finalMetrics: null,
        filesTouched: [],
        commands: [],
        allowedScopeOk: false,
        rollbackApplied: true,
        decision: "codex_failed",
        nextStep: status.candidate.next_step,
        hypothesis: status.candidate.hypothesis,
        whyThisChange: status.candidate.why_this_change,
        changesSummary: status.candidate.changes_summary,
        artifacts: status.candidate.artifacts,
      });
      await writeStatus(status);
      break;
    }

    const audit = auditFileChanges(codexResult.items, allowedDirs);
    const relevance = auditCommandRelevance(audit.filesTouched, smokeCommand, formalCommand);
    touchedFiles = audit.filesTouched;
    const observedCommands = collectCommandStrings(codexResult.items);
    const proposal = codexResult.proposal;

    let smokeMetrics: MetricSnapshot | null = null;
    let finalMetrics: MetricSnapshot | null = null;
    let rollbackApplied = false;
    let allowedScopeOk = audit.allowedScopeOk && relevance.relevantScopeOk;
    let decision = "reject";
    let nextStep = "换一个更小的改动，再跑 smoke。";
    let candidateArtifacts = [STATUS_PATH, TOOLS_LEDGER_PATH, MONITOR_LEDGER_PATH];

    if (!allowedScopeOk) {
      await restoreSnapshot(snapshot, touchedFiles);
      rollbackApplied = true;
      decision = relevance.relevantScopeOk ? "rollback_scope_violation" : "rollback_irrelevant_change";
      nextStep = relevance.relevantScopeOk
        ? "只保留允许目录内的文件改动。"
        : "改动必须直接作用于当前 smoke/formal 命令会用到的脚本或特征代码。";
      status.last_error = [...audit.violations, ...relevance.violations].join(", ");
    } else if (!dryRun) {
      const smokeCommandLine = ensureOutputJson(smokeCommand, smokeOutputPath);
      status.stage = "smoke";
      status.current_command = smokeCommandLine;
      status.updated_at = new Date().toISOString();
      await writeStatus(status);

      const smokeRun = await runShellCommand(smokeCommandLine, REPO_ROOT);
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
        decision = smokeRun.exitCode === 0 ? "reject_missing_metrics" : "reject_smoke_failed";
        nextStep = "先把 smoke 命令跑通，再继续。";
        status.last_error = smokeRun.exitCode === 0 ? "smoke metrics missing" : (smokeRun.stderr.trim() || smokeRun.stdout.trim() || "smoke command failed");
      } else if (isBetterThan(smokeMetrics.val_primary_metric, status.accepted_stable_best.val_primary_metric)) {
        status.stage = "formal_eval";
        const formalCommandLine = ensureOutputJson(formalCommand, formalOutputPath);
        status.current_command = formalCommandLine;
        status.updated_at = new Date().toISOString();
        await writeStatus(status);

        const formalRun = await runShellCommand(formalCommandLine, REPO_ROOT);
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
          decision = formalRun.exitCode === 0 ? "rollback_missing_formal_metrics" : "rollback_formal_failed";
          nextStep = "formal 没过，回到 smoke 再换一个候选。";
          status.last_error = formalRun.exitCode === 0 ? "formal metrics missing" : (formalRun.stderr.trim() || formalRun.stdout.trim() || "formal command failed");
        } else if (isBetterThan(finalMetrics.val_primary_metric, status.accepted_stable_best.val_primary_metric)) {
          await restoreSnapshot(snapshot, touchedFiles);
          rollbackApplied = true;
          status.leading_unverified_candidate = buildAcceptedBest({
            runId,
            primaryMetricName: smokeMetrics.primary_metric_name,
            smokeMetrics,
            formalMetrics: finalMetrics,
          });
          syncAcceptedBestAlias(status);
          decision = "hold_for_packet_gate";
          nextStep = "formal 超过 stable best，先记成未复验候选，再跑 packet gate。";
        } else {
          await restoreSnapshot(snapshot, touchedFiles);
          rollbackApplied = true;
          decision = "formal_not_better";
          nextStep = "formal 没超过当前最好结果，换更小的改动。";
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

    if (decision === "accept") {
      consecutiveFailures = 0;
      status.patience_streak = 0;
      status.stage = "accepted";
      status.last_error = null;
    } else if (decision === "hold_for_packet_gate") {
      consecutiveFailures = 0;
      status.patience_streak = 0;
      status.stage = "formal_eval";
      status.last_error = null;
    } else {
      consecutiveFailures += 1;
      status.patience_streak = consecutiveFailures;
      if (consecutiveFailures >= consecutiveLimit) {
        status.stage = "done";
        nextStep = "patience 用完，先停在当前最好结果。";
      } else {
        status.stage = rollbackApplied ? "rollback" : "rejected";
      }
    }

    status.current_command = "";
    status.candidate = {
      run_id: runId,
      stage: decision === "accept"
        ? "accepted"
        : decision === "hold_for_packet_gate"
          ? "formal_eval"
        : rollbackApplied
          ? "rollback"
          : finalMetrics
            ? "formal_eval"
            : smokeMetrics
              ? "smoke"
              : "baseline",
      hypothesis: proposal.hypothesis,
      why_this_change: proposal.why_this_change,
      changes_summary: proposal.changes_summary,
      files_touched: touchedFiles,
      commands: dedupeStrings(observedCommands),
      smoke_metrics: smokeMetrics,
      final_metrics: finalMetrics,
      allowed_scope_ok: allowedScopeOk,
      rollback_applied: rollbackApplied,
      decision,
      next_step: nextStep,
      artifacts: candidateArtifacts,
    };
    status.current_iteration = iteration;
    status.updated_at = new Date().toISOString();
    await recordIteration({
      status,
      iteration,
      runId,
      parentRunId,
      smokeMetrics,
      finalMetrics,
      filesTouched: touchedFiles,
      commands: dedupeStrings(observedCommands),
      allowedScopeOk,
      rollbackApplied,
      decision,
      nextStep,
      hypothesis: proposal.hypothesis,
      whyThisChange: proposal.why_this_change,
      changesSummary: proposal.changes_summary,
      artifacts: candidateArtifacts,
    });
    await writeStatus(status);

    if (status.stage === "done") {
      break;
    }
  }

  if (status.stage !== "done" && status.current_iteration >= status.max_iterations) {
    status.stage = "done";
    status.current_command = "";
    status.updated_at = new Date().toISOString();
    await writeStatus(status);
  }
}

async function initializeBaseline({
  status,
  baselineMetricsPath,
  baselineCommand,
  dryRun,
}: {
  status: CampaignStatus;
  baselineMetricsPath: string;
  baselineCommand: string;
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
    const result = await runShellCommand(ensureOutputJson(baselineCommand, outputPath), REPO_ROOT);
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
  status.stage = "smoke";
  status.current_command = baselineCommandUsed;
  status.candidate = buildEmptyCandidate(`${status.campaign_id}-baseline`);
  status.candidate.stage = "baseline";
  status.candidate.smoke_metrics = summary;
  status.candidate.final_metrics = null;
  status.candidate.allowed_scope_ok = true;
  status.candidate.rollback_applied = false;
  status.candidate.decision = "baseline_initialized";
  status.candidate.next_step = "开始 smoke 候选。";
  status.candidate.artifacts = baselineArtifacts;
  status.updated_at = new Date().toISOString();
  await writeStatus(status);

  await recordIteration({
    status,
    iteration: 0,
    runId: status.candidate.run_id,
    parentRunId: null,
    smokeMetrics: summary,
    finalMetrics: null,
    filesTouched: [],
    commands: baselineCommandUsed ? [baselineCommandUsed] : [],
    allowedScopeOk: true,
    rollbackApplied: false,
    decision: "baseline_initialized",
    nextStep: "开始 smoke 候选。",
    hypothesis: "用已有 smoke 结果初始化 campaign 基线。",
    whyThisChange: "先把当前最好结果写入 monitor，再开始后续 smoke/formal 迭代。",
    changesSummary: "初始化 baseline，确认 campaign 可以从现有 smoke 指标继续跑。",
    artifacts: baselineArtifacts,
  });
}

async function runBankQcGate({
  status,
  campaignId,
  runId,
  iteration,
  parentRunId,
  bankQcCommand,
  dryRun,
}: {
  status: CampaignStatus;
  campaignId: string;
  runId: string;
  iteration: number;
  parentRunId: string | null;
  bankQcCommand: string;
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
    decision: "bank_qc",
    next_step: "等待 bank-QC 结果。",
  };
  status.updated_at = new Date().toISOString();
  await writeStatus(status);

  const result = await runShellCommand(bankQcCommand, REPO_ROOT);
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
    hypothesis: "先确认当前 dataset 的 active_bank 和 raw channel half scan 一致。",
    why_this_change: "bank-QC 失败时，后续训练结果不再可比较。",
    changes_summary: "bank-QC 门禁失败，训练没有开始。",
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
  await recordIteration({
    status,
    iteration,
    runId,
    parentRunId,
    smokeMetrics: null,
    finalMetrics: null,
    filesTouched: [],
    commands: [result.command],
    allowedScopeOk: true,
    rollbackApplied: false,
    decision: "bank_qc_failed",
    nextStep: status.candidate.next_step,
    hypothesis: status.candidate.hypothesis,
    whyThisChange: status.candidate.why_this_change,
    changesSummary: status.candidate.changes_summary,
    artifacts: status.candidate.artifacts,
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
}: {
  campaignId: string;
  maxIterations: number;
  patience: number;
  existingStatus: Partial<CampaignStatus> | null;
  baselineMetricsPath: string;
  allowedDirs: string[];
}): CampaignStatus {
  if (existingStatus && existingStatus.campaign_id === campaignId) {
    return {
      campaign_id: campaignId,
      current_iteration: Number(existingStatus.current_iteration ?? 0),
      max_iterations: Number(existingStatus.max_iterations ?? maxIterations),
      patience: Number(existingStatus.patience ?? patience),
      stage: normalizeStage(existingStatus.stage),
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
      current_command: String(existingStatus.current_command ?? ""),
      updated_at: new Date().toISOString(),
      codex_thread_id: typeof existingStatus.codex_thread_id === "string" ? existingStatus.codex_thread_id : null,
      patience_streak: Number(existingStatus.patience_streak ?? 0),
      last_error: typeof existingStatus.last_error === "string" ? existingStatus.last_error : null,
    };
  }

  return {
    campaign_id: campaignId,
    current_iteration: 0,
    max_iterations: maxIterations,
    patience,
    stage: "baseline",
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
      test_primary_metric: null,
      test_rmse: null,
      feature_family: null,
      model_family: null,
      evaluation_mode: null,
      result_json: null,
      artifacts: [],
    },
    candidate: buildEmptyCandidate(`${campaignId}-bootstrap`),
    current_command: "",
    updated_at: new Date().toISOString(),
    codex_thread_id: null,
    patience_streak: 0,
    last_error: null,
  };
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
    hypothesis: String(record.hypothesis ?? ""),
    why_this_change: String(record.why_this_change ?? ""),
    changes_summary: String(record.changes_summary ?? ""),
    files_touched: parseStringArray(record.files_touched),
    commands: parseStringArray(record.commands),
    smoke_metrics: normalizeMetricSnapshot(record.smoke_metrics),
    final_metrics: normalizeMetricSnapshot(record.final_metrics),
    allowed_scope_ok: Boolean(record.allowed_scope_ok ?? false),
    rollback_applied: Boolean(record.rollback_applied ?? false),
    decision: String(record.decision ?? ""),
    next_step: String(record.next_step ?? ""),
    artifacts: parseStringArray(record.artifacts),
  };
}

function buildEmptyCandidate(runId: string): CandidateSnapshot {
  return {
    run_id: runId,
    stage: "baseline",
    hypothesis: "",
    why_this_change: "",
    changes_summary: "",
    files_touched: [],
    commands: [],
    smoke_metrics: null,
    final_metrics: null,
    allowed_scope_ok: false,
    rollback_applied: false,
    decision: "pending",
    next_step: "",
    artifacts: [],
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
  allowedDirs,
  smokeCommand,
  formalCommand,
}: {
  thread: ReturnType<Codex["startThread"]> | ReturnType<Codex["resumeThread"]>;
  campaignId: string;
  iteration: number;
  status: CampaignStatus;
  allowedDirs: string[];
  smokeCommand: string;
  formalCommand: string;
}): Promise<CodexEditResult> {
  const schema = {
    type: "object",
    properties: {
      hypothesis: { type: "string" },
      why_this_change: { type: "string" },
      changes_summary: { type: "string" },
      files_touched: { type: "array", items: { type: "string" } },
      next_step: { type: "string" },
    },
    required: ["hypothesis", "why_this_change", "changes_summary", "files_touched", "next_step"],
    additionalProperties: false,
  } as const;

  const prompt = [
    "你在维护一个受限 AutoResearch campaign。",
    "只允许修改这些目录内的文件：",
    ...allowedDirs.map((dir) => `- ${dir}`),
    "",
    "绝对不要改这些内容：",
    "- split",
    "- 对齐",
    "- primary metric",
    "- 原始读取边界",
    "- scripts/convert_session.py",
    "- src/bci_autoresearch/data/**",
    "",
    "这轮只做一处小改动，目标是让 smoke 指标更可能上升。",
    "改动必须直接作用于当前 smoke/formal 命令会实际执行到的训练脚本、模型或特征代码。",
    "不要运行训练、评估或格式化命令，外部执行器会做校验。",
    "完成后只返回 JSON，不要加多余说明。",
    "",
    `campaign_id: ${campaignId}`,
    `iteration: ${iteration}`,
    `current_stage: ${status.stage}`,
    `current_best_metric: ${status.accepted_stable_best.val_primary_metric ?? "null"}`,
    `current_best_source: ${status.accepted_stable_best.artifacts[0] ?? ""}`,
    `smoke_command: ${smokeCommand}`,
    `formal_command: ${formalCommand}`,
    "",
    "当前状态：",
    JSON.stringify(status, null, 2),
  ].join("\n");

  const turn = await thread.run(prompt, { outputSchema: schema });
  const proposal = parseProposal(turn.finalResponse);
  return {
    proposal,
    items: turn.items,
    threadId: thread.id ?? null,
  };
}

function parseProposal(text: string): {
  hypothesis: string;
  why_this_change: string;
  changes_summary: string;
  files_touched: string[];
  next_step: string;
} {
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    return {
      hypothesis: String(parsed.hypothesis ?? ""),
      why_this_change: String(parsed.why_this_change ?? ""),
      changes_summary: String(parsed.changes_summary ?? ""),
      files_touched: parseStringArray(parsed.files_touched),
      next_step: String(parsed.next_step ?? ""),
    };
  } catch {
    return {
      hypothesis: text.trim(),
      why_this_change: "",
      changes_summary: "",
      files_touched: [],
      next_step: "",
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

function auditCommandRelevance(filesTouched: string[], smokeCommand: string, formalCommand: string): RelevanceAudit {
  const relevantPatterns = deriveRelevantPatterns(smokeCommand, formalCommand);
  if (filesTouched.length === 0) {
    return {
      relevantScopeOk: false,
      relevantPatterns,
      violations: ["no_files_touched"],
    };
  }

  const violations = filesTouched.filter((item) => !relevantPatterns.some((pattern) => new RegExp(pattern).test(item)));
  return {
    relevantScopeOk: violations.length === 0,
    relevantPatterns,
    violations,
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
  return allowedDirs.some((dir) => {
    const normalizedDir = normalizeRepoPath(dir);
    return normalized === normalizedDir || normalized.startsWith(`${normalizedDir}/`);
  });
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

async function runShellCommand(command: string, cwd: string): Promise<RunCommandResult> {
  const startedAt = Date.now();
  const shell = process.platform === "win32" ? "cmd.exe" : process.env.SHELL ?? "/bin/zsh";
  const args = process.platform === "win32" ? ["/c", command] : ["-lc", command];

  return await new Promise<RunCommandResult>((resolve, reject) => {
    const child = spawn(shell, args, {
      cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (exitCode) => {
      resolve({
        command,
        exitCode: exitCode ?? -1,
        stdout,
        stderr,
        durationMs: Date.now() - startedAt,
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
  smokeMetrics,
  finalMetrics,
  filesTouched,
  commands,
  allowedScopeOk,
  rollbackApplied,
  decision,
  nextStep,
  hypothesis,
  whyThisChange,
  changesSummary,
  artifacts,
}: {
  status: CampaignStatus;
  iteration: number;
  runId: string;
  parentRunId: string | null;
  smokeMetrics: MetricSnapshot | null;
  finalMetrics: MetricSnapshot | null;
  filesTouched: string[];
  commands: string[];
  allowedScopeOk: boolean;
  rollbackApplied: boolean;
  decision: string;
  nextStep: string;
  hypothesis: string;
  whyThisChange: string;
  changesSummary: string;
  artifacts: string[];
}) {
  const record: CampaignRecord = {
    campaign_id: status.campaign_id,
    run_id: runId,
    parent_run_id: parentRunId,
    iteration,
    stage: status.stage,
    recorded_at: new Date().toISOString(),
    agent_name: DEFAULT_AGENT_NAME,
    dataset_name: status.accepted_stable_best.dataset_name,
    target_mode: status.accepted_stable_best.target_mode,
    target_space: status.accepted_stable_best.target_space,
    primary_metric_name: status.accepted_stable_best.primary_metric_name || DEFAULT_PRIMARY_METRIC_NAME,
    hypothesis,
    why_this_change: whyThisChange,
    changes_summary: changesSummary,
    files_touched: filesTouched,
    commands,
    smoke_metrics: smokeMetrics,
    final_metrics: finalMetrics,
    allowed_scope_ok: allowedScopeOk,
    rollback_applied: rollbackApplied,
    decision,
    next_step: nextStep,
    artifacts,
  };

  const line = `${JSON.stringify(record)}\n`;
  await writeFile(TOOLS_LEDGER_PATH, line, { flag: "a" });
  await writeFile(MONITOR_LEDGER_PATH, line, { flag: "a" });
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
    test_primary_metric: testPrimary,
    test_rmse: extractNumber(metrics, "test_metrics.mean_rmse_macro", "test_rmse"),
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

async function writeStatus(status: CampaignStatus) {
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

main().catch((error) => {
  process.stderr.write(`${stringifyError(error)}\n`);
  process.exitCode = 1;
});
