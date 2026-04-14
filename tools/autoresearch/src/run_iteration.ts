import { randomUUID } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Codex } from "@openai/codex-sdk";

type Primitive = string | number | boolean | null;
type JsonValue = Primitive | JsonValue[] | { [key: string]: JsonValue };

interface IterationRecord {
  run_id: string;
  parent_run_id: string | null;
  recorded_at: string;
  agent_name: string;
  target_mode: string;
  hypothesis: string;
  why_this_change: string;
  changes_summary: string;
  files_touched: string[];
  commands: string[];
  dataset_name: string;
  split_sessions: JsonValue;
  metrics: JsonValue;
  decision: string;
  next_step: string;
  artifacts: string[];
}

interface ParsedArgs {
  [key: string]: string | boolean;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TOOLS_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(TOOLS_ROOT, "..", "..");

const STATUS_PATH = path.join(REPO_ROOT, "memory", "2026-04-05_status.md");
const LEDGER_PATH = path.join(REPO_ROOT, "artifacts", "monitor", "experiment_ledger.jsonl");
const OUTPUT_PATH = path.join(TOOLS_ROOT, "experiment_ledger.jsonl");

const DEFAULT_DATASET_NAME = "walk_matched_v1_64clean";
const DEFAULT_SPLIT_SESSIONS = { train: 18, val: 2, test: 2 };
const DEFAULT_AGENT_NAME = "restricted-autoresearch-codex-sdk";
const DEFAULT_TARGET_MODE = "restricted-autoresearch";
const AUTORESEARCH_PLANNER_REASONING_EFFORT = "medium";

const ALLOWLIST = [
  /^scripts\/train_[^/]+\.py$/,
  /^src\/bci_autoresearch\/models\/.+$/,
  /^src\/bci_autoresearch\/features\/.+$/,
  /^scripts\/build_monitor_artifacts\.py$/,
  /^scripts\/analyze_channel_halves\.py$/,
];

const DENYLIST_PATHS = [
  /^scripts\/convert_session\.py$/,
  /^src\/bci_autoresearch\/data\/.+$/,
  /^data\/.+$/,
  /^\/Volumes\/.+$/,
];

const DENYLIST_COMMAND_TERMS = [
  "--split",
  "--split-",
  "--split_",
  " split ",
  "split=",
  "split_sessions",
  "split-session",
  "split_session",
  "alignment",
  "align=",
  "aligned",
  "align ",
  " align",
  "primary-metric",
  "primary metric",
  "primary_metric",
  "convert_session.py",
];

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const statusText = await readTextIfExists(STATUS_PATH);
  const ledgerText = await readTextIfExists(LEDGER_PATH);

  const parentRunId = getLastRunId(ledgerText);
  const runId = String(args["run-id"] ?? args["run_id"] ?? `autoresearch-${Date.now()}-${randomUUID().slice(0, 8)}`);
  const recordedAt = new Date().toISOString();

  const filesTouched = parseStringList(args["files-touched"] ?? args["files_touched"], []);
  validateTouchedFiles(filesTouched);

  const commands = parseStringList(args["commands"], []);
  validateCommands(commands);

  const record: IterationRecord = {
    run_id: runId,
    parent_run_id:
      String(args["parent-run-id"] ?? args["parent_run_id"] ?? "").trim() ||
      parentRunId,
    recorded_at: recordedAt,
    agent_name: String(args["agent-name"] ?? args["agent_name"] ?? DEFAULT_AGENT_NAME),
    target_mode: String(args["target-mode"] ?? args["target_mode"] ?? DEFAULT_TARGET_MODE),
    hypothesis:
      String(args["hypothesis"] ?? "").trim() ||
      buildDefaultText("hypothesis", statusText),
    why_this_change:
      String(args["why-this-change"] ?? args["why_this_change"] ?? "").trim() ||
      "把 AutoResearch 的活动范围收紧到允许的训练脚本、模型和特征代码，再把迭代记录固定成 JSONL，避免碰 split、对齐、primary metric、数据读取和原始路径。",
    changes_summary:
      String(args["changes-summary"] ?? args["changes_summary"] ?? "").trim() ||
      "新增受限 AutoResearch 脚手架，含 Codex SDK TypeScript 入口、目录约束、ledger 读取和迭代记录输出。",
    files_touched: filesTouched,
    commands,
    dataset_name: String(args["dataset-name"] ?? args["dataset_name"] ?? DEFAULT_DATASET_NAME),
    split_sessions: parseJsonArg(args["split-sessions"] ?? args["split_sessions"], DEFAULT_SPLIT_SESSIONS),
    metrics: parseJsonArg(args["metrics"], {}),
    decision: String(args["decision"] ?? "continue"),
    next_step:
      String(args["next-step"] ?? args["next_step"] ?? "").trim() ||
      "先在允许范围内跑第一轮模型或特征改动，再只记录结果，不放开 split、对齐、primary metric 和原始数据读取。",
    artifacts: parseStringList(args["artifacts"], []),
  };

  const agentSummary = await maybeRunCodexPlanner(record, statusText);
  if (agentSummary) {
    record.changes_summary = `${record.changes_summary} | planner: ${agentSummary}`;
  }

  const line = `${JSON.stringify(record)}\n`;
  await mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await writeFile(OUTPUT_PATH, line, { flag: "a" });
  process.stdout.write(line);
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

async function readTextIfExists(filePath: string): Promise<string> {
  try {
    return await readFile(filePath, "utf8");
  } catch {
    return "";
  }
}

function getLastRunId(ledgerText: string): string | null {
  const lines = ledgerText.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return null;
  }

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const entry = JSON.parse(lines[index]) as { run_id?: unknown };
      if (typeof entry.run_id === "string" && entry.run_id.trim()) {
        return entry.run_id;
      }
    } catch {
      continue;
    }
  }

  return null;
}

function parseStringList(value: unknown, fallback: string[]): string[] {
  if (typeof value !== "string" || value.trim() === "") {
    return fallback;
  }

  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseJsonArg(value: unknown, fallback: JsonValue): JsonValue {
  if (typeof value !== "string" || value.trim() === "") {
    return fallback;
  }

  try {
    return JSON.parse(value) as JsonValue;
  } catch {
    return fallback;
  }
}

function validateTouchedFiles(files: string[]) {
  for (const file of files) {
    const normalized = toRepoRelativePath(file);
    const allowed = ALLOWLIST.some((pattern) => pattern.test(normalized));
    const denied = DENYLIST_PATHS.some((pattern) => pattern.test(normalized));

    if (denied) {
      throw new Error(`blocked file path: ${file}`);
    }

    if (!allowed) {
      throw new Error(`path outside allowlist: ${file}`);
    }
  }
}

function validateCommands(commands: string[]) {
  for (const command of commands) {
    const normalized = command.toLowerCase();
    for (const term of DENYLIST_COMMAND_TERMS) {
      if (normalized.includes(term.toLowerCase())) {
        throw new Error(`blocked command term "${term}" in: ${command}`);
      }
    }
  }
}

function toRepoRelativePath(input: string): string {
  const candidate = path.isAbsolute(input) ? path.relative(REPO_ROOT, input) : input;
  return candidate.split(path.sep).join("/");
}

function buildDefaultText(kind: "hypothesis", statusText: string): string {
  if (kind === "hypothesis") {
    return [
      "在只允许训练脚本、模型和特征代码的前提下，先把最小可控改动做清楚，",
      "再看监控工件是否比上一轮更稳定。",
      statusText.includes("walk_matched_v1_64clean")
        ? "当前基线仍是 walk_matched_v1_64clean，所以下一轮优先从允许范围里找更稳的表示。"
        : "",
    ]
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  }

  return "";
}

async function maybeRunCodexPlanner(record: IterationRecord, statusText: string): Promise<string | null> {
  if (process.env.AUTORESEARCH_USE_CODEX !== "1") {
    return null;
  }

  const codex = new Codex({
    config: {
      model_reasoning_effort: AUTORESEARCH_PLANNER_REASONING_EFFORT,
    },
  });
  const priorThreadId = process.env.AUTORESEARCH_THREAD_ID?.trim();
  const thread = priorThreadId ? codex.resumeThread(priorThreadId) : codex.startThread();
  const prompt = [
    "你是受限 AutoResearch 规划器。",
    "只能提出训练脚本、模型脚本、特征脚本相关的下一步建议。",
    "绝对不要建议修改 split、对齐、primary metric、convert_session.py、原始路径。",
    "请只返回一句中文短句，描述下一步最安全、最值得验证的改动。",
    "",
    "项目状态：",
    statusText,
    "",
    "当前迭代记录草稿：",
    JSON.stringify(record, null, 2),
  ].join("\n");
  const result = await thread.run(prompt);
  const text = summarizeCodexResult(result);
  return text ? text.trim() : null;
}

function summarizeCodexResult(result: unknown): string {
  if (typeof result === "string") {
    return result;
  }
  if (result && typeof result === "object") {
    const asRecord = result as Record<string, unknown>;
    for (const key of ["outputText", "text", "finalOutput"]) {
      const value = asRecord[key];
      if (typeof value === "string" && value.trim()) {
        return value;
      }
    }
  }
  return "";
}

await main();
