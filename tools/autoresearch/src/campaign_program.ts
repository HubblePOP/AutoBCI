import { readFile } from "node:fs/promises";
import path from "node:path";

export interface ProgramDocuments {
  constitutionPath: string;
  constitutionText: string;
  derivedProgramPath: string;
  derivedProgramText: string;
  currentProgramPath: string;
  currentProgramText: string;
}

export interface CampaignTrack {
  trackId: string;
  topicId?: string;
  runnerFamily?: string;
  expectedMemoryClass?: "high" | "low" | "service";
  internetResearchEnabled?: boolean;
  validated?: boolean;
  skipCodexEdit?: boolean;
  trackGoal: string;
  promotionTarget: string;
  smokeCommand: string;
  formalCommand: string;
  allowedChangeScope: string[];
}

export interface CampaignManifest {
  manifestPath: string;
  reviewCadence: string;
  tracks: CampaignTrack[];
}

export interface CampaignPromptStatusSnapshot {
  stage: string;
  acceptedStableBestMetric: number | null;
  acceptedStableBestSource: string;
  canonicalPromotionMetric: string;
}

export interface CampaignRecordLike {
  campaign_id: string;
  run_id: string;
  recorded_at: string;
  track_id: string;
  track_goal: string;
  decision: string;
  change_bucket: string;
  track_comparison_note: string;
  smoke_metrics: {
    val_primary_metric: number | null;
    test_primary_metric: number | null;
  } | null;
  final_metrics: {
    val_primary_metric: number | null;
    test_primary_metric: number | null;
  } | null;
}

export interface DailyReviewTrackSummary {
  trackId: string;
  trackGoal: string;
  runCount: number;
  latestDecision: string;
  latestRunId: string;
  latestChangeBucket: string;
  latestComparisonNote: string;
  bestValMetric: number | null;
  bestTestMetric: number | null;
}

export interface DailyReviewPromotionCandidate {
  trackId: string;
  runId: string;
  decision: string;
  comparisonNote: string;
  valMetric: number | null;
  testMetric: number | null;
}

export interface DailyReviewPacket {
  campaignId: string;
  reviewDate: string;
  generatedAt: string;
  trackSummaries: DailyReviewTrackSummary[];
  promotionCandidates: DailyReviewPromotionCandidate[];
}

export interface TrackOutcomeDecision {
  decision: string;
  updateLocalBest: boolean;
  updateGlobalCandidate: boolean;
}

export async function loadProgramDocuments(toolsRoot: string): Promise<ProgramDocuments> {
  const repoRoot = path.resolve(toolsRoot, "..", "..");
  const constitutionPath = path.join(repoRoot, "docs", "CONSTITUTION.md");
  const derivedProgramPath = path.join(toolsRoot, "program.md");
  const currentProgramPath = path.join(toolsRoot, "program.current.md");

  const [constitutionText, derivedProgramText, currentProgramText] = await Promise.all([
    readRequiredText(constitutionPath, "docs/CONSTITUTION.md"),
    readRequiredText(derivedProgramPath, "program.md"),
    readRequiredText(currentProgramPath, "program.current.md"),
  ]);

  return {
    constitutionPath,
    constitutionText,
    derivedProgramPath,
    derivedProgramText,
    currentProgramPath,
    currentProgramText,
  };
}

export async function loadTrackManifest(
  manifestPath: string,
  options: { defaultAllowedChangeScope: string[] },
): Promise<CampaignManifest> {
  const raw = JSON.parse(await readRequiredText(manifestPath, path.basename(manifestPath))) as Record<string, unknown>;
  const reviewCadence = typeof raw.review_cadence === "string" && raw.review_cadence.trim() !== ""
    ? raw.review_cadence
    : "daily";

  const rawTracks = Array.isArray(raw.tracks) ? raw.tracks : [];
  if (rawTracks.length === 0) {
    throw new Error(`Track manifest ${manifestPath} must define at least one track.`);
  }

  const tracks = rawTracks.map((item, index) => normalizeTrack(item, index, options.defaultAllowedChangeScope));
  return {
    manifestPath,
    reviewCadence,
    tracks,
  };
}

export function buildCodexPrompt({
  campaignId,
  iteration,
  allowedDirs,
  programDocuments,
  track,
  statusSnapshot,
  promptMode,
}: {
  campaignId: string;
  iteration: number;
  allowedDirs: string[];
  programDocuments: ProgramDocuments;
  track: CampaignTrack;
  statusSnapshot: CampaignPromptStatusSnapshot;
  promptMode?: "exploration" | "closeout";
}): string {
  if (promptMode === "closeout") {
    return [
      "你在维护一个受限 AutoResearch campaign，当前已经进入收尾模式。",
      "这轮只允许做最小必要动作，用来补 smoke/formal、完成记账，或明确撤回原因。",
      "不要重新展开大范围研究，也不要为低优先级分支重新开搜索回合。",
      "",
      "当前 track：",
      `- track_id: ${track.trackId}`,
      ...(track.topicId ? [`- topic_id: ${track.topicId}`] : []),
      ...(track.runnerFamily ? [`- runner_family: ${track.runnerFamily}`] : []),
      `- track_goal: ${track.trackGoal}`,
      `- promotion_target: ${track.promotionTarget}`,
      `- smoke_command: ${track.smokeCommand}`,
      `- formal_command: ${track.formalCommand}`,
      "",
      "最近一次结果：",
      `- current_stage: ${statusSnapshot.stage}`,
      `- current_best_metric: ${statusSnapshot.acceptedStableBestMetric ?? "null"}`,
      `- current_best_source: ${statusSnapshot.acceptedStableBestSource}`,
      `- canonical_promotion_metric: ${statusSnapshot.canonicalPromotionMetric}`,
      "",
      "绝对不要改这些内容：",
      "- split",
      "- 对齐",
      "- primary metric",
      "- 原始读取边界",
      "- scripts/convert_session.py",
      "- src/bci_autoresearch/data/**",
      "",
      "这轮允许的最小动作：",
      "- 只做能直接服务当前 track 比较的最小改动",
      "- 不要重新贴研究背景",
      "- 不要重新开联网搜索",
      "",
      "输出 JSON 必须包含：",
      "- hypothesis",
      "- why_this_change",
      "- changes_summary",
      "- files_touched",
      "- next_step",
      "- change_bucket  (必须是 representation-led 或 model-led 之一)",
      "- track_comparison_note",
      "- search_queries  (收尾模式默认返回空数组)",
      "- research_evidence  (收尾模式默认返回空数组)",
      "",
      `campaign_id: ${campaignId}`,
      `iteration: ${iteration}`,
      `allowed_change_scope: ${track.allowedChangeScope.join(", ")}`,
      `allowed_dirs: ${allowedDirs.join(", ")}`,
    ].join("\n");
  }

  return [
    "你在维护一个受限 AutoResearch campaign。",
    "先阅读仓库总纲、执行派生契约和当前附录，再结合当前 track 做一处小改动。",
    "",
    `CONSTITUTION.md: ${programDocuments.constitutionPath}`,
    summarizePromptDocument(programDocuments.constitutionText),
    "",
    `program.md: ${programDocuments.derivedProgramPath}`,
    summarizePromptDocument(programDocuments.derivedProgramText),
    "",
    `program.current.md: ${programDocuments.currentProgramPath}`,
    summarizePromptDocument(programDocuments.currentProgramText),
    "",
    "当前 track：",
    `- track_id: ${track.trackId}`,
    ...(track.topicId ? [`- topic_id: ${track.topicId}`] : []),
    ...(track.runnerFamily ? [`- runner_family: ${track.runnerFamily}`] : []),
    ...(track.expectedMemoryClass ? [`- expected_memory_class: ${track.expectedMemoryClass}`] : []),
    `- internet_research_enabled: ${track.internetResearchEnabled ? "true" : "false"}`,
    `- track_goal: ${track.trackGoal}`,
    `- promotion_target: ${track.promotionTarget}`,
    `- smoke_command: ${track.smokeCommand}`,
    `- formal_command: ${track.formalCommand}`,
    `- allowed_change_scope: ${track.allowedChangeScope.join(", ")}`,
    "",
    "全局允许目录：",
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
    "输出 JSON 必须包含：",
    "- hypothesis",
    "- why_this_change",
    "- changes_summary",
    "- files_touched",
    "- next_step",
    "- change_bucket  (必须是 representation-led 或 model-led 之一)",
    "- track_comparison_note",
    "- search_queries  (如果你没有联网搜索，就返回空数组；如果你要联网搜索，必须逐条写出原始查询词和意图)",
    "- research_evidence  (如果你引用了论文、帖子、GitHub 或文档，必须逐条写出来源、标题、URL 和 why_it_matters)",
    "",
    "如果你要联网搜索：",
    "- 只把它当作候选灵感和支持材料，不要把外部来源当作自动通过证据。",
    "- 你搜了什么、看了什么，最后都必须进 output JSON。",
    "",
    `campaign_id: ${campaignId}`,
    `iteration: ${iteration}`,
    `current_stage: ${statusSnapshot.stage}`,
    `current_best_metric: ${statusSnapshot.acceptedStableBestMetric ?? "null"}`,
    `current_best_source: ${statusSnapshot.acceptedStableBestSource}`,
    `canonical_promotion_metric: ${statusSnapshot.canonicalPromotionMetric}`,
  ].join("\n");
}

function summarizePromptDocument(text: string, options?: { maxLines?: number; maxChars?: number }): string {
  const maxLines = options?.maxLines ?? 18;
  const maxChars = options?.maxChars ?? 1800;
  const normalized = text
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter((line, index, lines) => line !== "" || (index > 0 && lines[index - 1] !== ""));
  const clippedLines = normalized.slice(0, maxLines);
  const joined = clippedLines.join("\n").trim();
  if (joined.length <= maxChars && normalized.length <= maxLines) {
    return joined;
  }
  return `${joined.slice(0, maxChars).trimEnd()}\n\n[truncated for prompt efficiency]`;
}

export function buildDailyReviewPacket({
  campaignId,
  reviewDate,
  records,
}: {
  campaignId: string;
  reviewDate: string;
  records: CampaignRecordLike[];
}): DailyReviewPacket {
  const filtered = records
    .filter((record) => record.campaign_id === campaignId)
    .filter((record) => record.recorded_at.startsWith(reviewDate))
    .sort((left, right) => left.recorded_at.localeCompare(right.recorded_at));

  const byTrack = new Map<string, CampaignRecordLike[]>();
  for (const record of filtered) {
    const rows = byTrack.get(record.track_id) ?? [];
    rows.push(record);
    byTrack.set(record.track_id, rows);
  }

  const trackSummaries = Array.from(byTrack.entries()).map(([trackId, rows]) => {
    const latest = rows[rows.length - 1];
    const bestValMetric = maxMetric(rows, "val");
    const bestTestMetric = maxMetric(rows, "test");
    return {
      trackId,
      trackGoal: latest?.track_goal ?? "",
      runCount: rows.length,
      latestDecision: latest?.decision ?? "",
      latestRunId: latest?.run_id ?? "",
      latestChangeBucket: latest?.change_bucket ?? "",
      latestComparisonNote: latest?.track_comparison_note ?? "",
      bestValMetric,
      bestTestMetric,
    };
  });

  const promotionCandidates = filtered
    .filter((record) => record.decision === "hold_for_promotion_review" || record.decision === "hold_for_packet_gate")
    .map((record) => ({
      trackId: record.track_id,
      runId: record.run_id,
      decision: record.decision,
      comparisonNote: record.track_comparison_note,
      valMetric: record.final_metrics?.val_primary_metric ?? record.smoke_metrics?.val_primary_metric ?? null,
      testMetric: record.final_metrics?.test_primary_metric ?? record.smoke_metrics?.test_primary_metric ?? null,
    }));

  return {
    campaignId,
    reviewDate,
    generatedAt: `${reviewDate}T00:00:00.000Z`,
    trackSummaries,
    promotionCandidates,
  };
}

export function renderDailyReviewPacketMarkdown(packet: DailyReviewPacket): string {
  const lines = [
    `# AutoResearch Daily Review Packet`,
    "",
    `- campaign_id: ${packet.campaignId}`,
    `- review_date: ${packet.reviewDate}`,
    `- generated_at: ${packet.generatedAt}`,
    "",
    "## Track Summaries",
    "",
  ];

  if (packet.trackSummaries.length === 0) {
    lines.push("- No track activity recorded.");
  } else {
    for (const summary of packet.trackSummaries) {
      lines.push(
        `- ${summary.trackId}: runs=${summary.runCount}, latest_decision=${summary.latestDecision}, latest_run=${summary.latestRunId}, latest_bucket=${summary.latestChangeBucket}, best_val=${formatMetric(summary.bestValMetric)}, best_test=${formatMetric(summary.bestTestMetric)}`,
      );
      lines.push(`  note: ${summary.latestComparisonNote}`);
      lines.push(`  goal: ${summary.trackGoal}`);
    }
  }

  lines.push("", "## Promotion Candidates", "");
  if (packet.promotionCandidates.length === 0) {
    lines.push("- None.");
  } else {
    for (const candidate of packet.promotionCandidates) {
      lines.push(
        `- ${candidate.trackId} / ${candidate.runId}: decision=${candidate.decision}, val=${formatMetric(candidate.valMetric)}, test=${formatMetric(candidate.testMetric)}`,
      );
      lines.push(`  note: ${candidate.comparisonNote}`);
    }
  }

  return lines.join("\n");
}

export function decideTrackOutcome({
  promotionTarget,
  formalValMetric,
  localBestMetric,
  globalStableBestMetric,
  canonicalRetestReady,
}: {
  promotionTarget: string;
  formalValMetric: number | null;
  localBestMetric: number | null;
  globalStableBestMetric: number | null;
  canonicalRetestReady: boolean;
}): TrackOutcomeDecision {
  if (formalValMetric === null || !isBetterThan(formalValMetric, localBestMetric)) {
    return {
      decision: "formal_not_better",
      updateLocalBest: false,
      updateGlobalCandidate: false,
    };
  }

  if (promotionTarget === "canonical_mainline" && canonicalRetestReady && isBetterThan(formalValMetric, globalStableBestMetric)) {
    return {
      decision: "hold_for_packet_gate",
      updateLocalBest: true,
      updateGlobalCandidate: true,
    };
  }

  return {
    decision: "hold_for_promotion_review",
    updateLocalBest: true,
    updateGlobalCandidate: false,
  };
}

async function readRequiredText(filePath: string, label: string): Promise<string> {
  try {
    return await readFile(filePath, "utf8");
  } catch (error) {
    throw new Error(`Missing required AutoResearch file: ${label} (${filePath})`, {
      cause: error,
    });
  }
}

function normalizeTrack(
  value: unknown,
  index: number,
  defaultAllowedChangeScope: string[],
): CampaignTrack {
  if (!value || typeof value !== "object") {
    throw new Error(`Track manifest entry ${index} is not an object.`);
  }
  const record = value as Record<string, unknown>;
  const trackId = readRequiredString(record.track_id, `tracks[${index}].track_id`);
  const topicId = readOptionalString(record.topic_id);
  const runnerFamily = readOptionalString(record.runner_family);
  const expectedMemoryClass = normalizeExpectedMemoryClass(
    readOptionalString(record.expected_memory_class),
    runnerFamily,
  );
  const internetResearchEnabled = typeof record.internet_research_enabled === "boolean"
    ? record.internet_research_enabled
    : false;
  const validated = typeof record.validated === "boolean" ? record.validated : false;
  const skipCodexEdit = typeof record.skip_codex_edit === "boolean"
    ? record.skip_codex_edit
    : validated;
  const trackGoal = readRequiredString(record.track_goal, `tracks[${index}].track_goal`);
  const promotionTarget = readRequiredString(record.promotion_target, `tracks[${index}].promotion_target`);
  const smokeCommand = readRequiredString(record.smoke_command, `tracks[${index}].smoke_command`);
  const formalCommand = readRequiredString(record.formal_command, `tracks[${index}].formal_command`);
  const allowedChangeScope = parseStringArray(record.allowed_change_scope);

  return {
    trackId,
    topicId,
    runnerFamily,
    expectedMemoryClass,
    internetResearchEnabled,
    validated,
    skipCodexEdit,
    trackGoal,
    promotionTarget,
    smokeCommand,
    formalCommand,
    allowedChangeScope: allowedChangeScope.length > 0 ? allowedChangeScope : [...defaultAllowedChangeScope],
  };
}

function readRequiredString(value: unknown, label: string): string {
  if (typeof value === "string" && value.trim() !== "") {
    return value.trim();
  }
  throw new Error(`Missing required field ${label}.`);
}

function readOptionalString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed === "" ? undefined : trimmed;
}

function normalizeExpectedMemoryClass(
  explicitValue: string | undefined,
  runnerFamily: string | undefined,
): "high" | "low" | "service" {
  if (explicitValue === "high" || explicitValue === "low" || explicitValue === "service") {
    return explicitValue;
  }

  const normalizedRunner = (runnerFamily || "").toLowerCase();
  if (
    normalizedRunner === "feature_lstm"
    || normalizedRunner === "feature_gru"
    || normalizedRunner === "feature_tcn"
    || normalizedRunner === "feature_cnn_lstm"
    || normalizedRunner === "feature_state_space_lite"
    || normalizedRunner === "feature_conformer_lite"
    || normalizedRunner === "raw_lstm"
  ) {
    return "high";
  }

  return "low";
}

function parseStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string" && item.trim() !== "");
}

function maxMetric(records: CampaignRecordLike[], kind: "val" | "test"): number | null {
  let best: number | null = null;
  for (const record of records) {
    const value = kind === "val"
      ? record.final_metrics?.val_primary_metric ?? record.smoke_metrics?.val_primary_metric ?? null
      : record.final_metrics?.test_primary_metric ?? record.smoke_metrics?.test_primary_metric ?? null;
    if (value === null) {
      continue;
    }
    if (best === null || value > best) {
      best = value;
    }
  }
  return best;
}

function formatMetric(value: number | null): string {
  if (value === null) {
    return "-";
  }
  return value.toFixed(4);
}

function isBetterThan(candidate: number | null, current: number | null): boolean {
  if (candidate === null) {
    return false;
  }
  if (current === null) {
    return true;
  }
  return candidate > current + 1e-9;
}
