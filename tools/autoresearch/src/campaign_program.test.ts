import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  buildCodexPrompt,
  buildDailyReviewPacket,
  decideTrackOutcome,
  loadProgramDocuments,
  loadTrackManifest,
  renderDailyReviewPacketMarkdown,
  type CampaignRecordLike,
} from "./campaign_program.js";

test("loadProgramDocuments reads both program layers from the tools root", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-program-"));
  await mkdir(path.join(tempRoot, "docs"), { recursive: true });
  const toolsRoot = path.join(tempRoot, "tools", "autoresearch");
  await mkdir(toolsRoot, { recursive: true });
  await writeFile(path.join(tempRoot, "docs", "CONSTITUTION.md"), "# repo constitution\n", "utf8");
  await writeFile(path.join(toolsRoot, "program.md"), "# execution contract\n", "utf8");
  await writeFile(path.join(toolsRoot, "program.current.md"), "# current campaign\n", "utf8");

  try {
    const docs = await loadProgramDocuments(toolsRoot);
    assert.equal(docs.constitutionPath, path.join(tempRoot, "docs", "CONSTITUTION.md"));
    assert.equal(docs.derivedProgramPath, path.join(toolsRoot, "program.md"));
    assert.equal(docs.currentProgramPath, path.join(toolsRoot, "program.current.md"));
    assert.match(docs.constitutionText, /repo constitution/);
    assert.match(docs.derivedProgramText, /execution contract/);
    assert.match(docs.currentProgramText, /current campaign/);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("loadProgramDocuments fails loudly when docs/CONSTITUTION.md is missing", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-program-missing-"));
  const toolsRoot = path.join(tempRoot, "tools", "autoresearch");
  await mkdir(toolsRoot, { recursive: true });
  await writeFile(path.join(toolsRoot, "program.md"), "# execution contract\n", "utf8");
  await writeFile(path.join(toolsRoot, "program.current.md"), "# current campaign\n", "utf8");

  try {
    await assert.rejects(
      loadProgramDocuments(toolsRoot),
      /CONSTITUTION\.md/,
    );
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("loadProgramDocuments fails loudly when program.current.md is missing", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-program-missing-current-"));
  const toolsRoot = path.join(tempRoot, "tools", "autoresearch");
  await mkdir(path.join(tempRoot, "docs"), { recursive: true });
  await mkdir(toolsRoot, { recursive: true });
  await writeFile(path.join(tempRoot, "docs", "CONSTITUTION.md"), "# repo constitution\n", "utf8");
  await writeFile(path.join(toolsRoot, "program.md"), "# execution contract\n", "utf8");

  try {
    await assert.rejects(
      loadProgramDocuments(toolsRoot),
      /program\.current\.md/,
    );
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("loadTrackManifest parses stable track definitions and applies fallback scopes", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-manifest-"));
  const manifestPath = path.join(tempRoot, "tracks.json");
  await writeFile(
    manifestPath,
    JSON.stringify(
      {
        review_cadence: "daily",
        tracks: [
          {
            track_id: "canonical_mainline",
            topic_id: "canonical_mainline",
            runner_family: "tree_xgboost",
            internet_research_enabled: true,
            track_goal: "Protect the canonical joints mainline.",
            promotion_target: "canonical_mainline",
            smoke_command: "python scripts/train_lstm.py --dataset-config joints_smoke.yaml",
            formal_command: "python scripts/train_lstm.py --dataset-config joints.yaml",
          },
          {
            track_id: "relative_origin_xyz",
            topic_id: "relative_origin_xyz",
            runner_family: "tree_xgboost",
            track_goal: "Explore RSCA-relative xyz targets.",
            promotion_target: "canonical_mainline",
            smoke_command: "python scripts/train_tree_baseline.py --dataset-config rel_smoke.yaml --model-family xgboost",
            formal_command: "python scripts/train_tree_baseline.py --dataset-config rel.yaml --model-family xgboost",
            allowed_change_scope: ["src/bci_autoresearch/features"],
          },
        ],
      },
      null,
      2,
    ),
    "utf8",
  );

  try {
    const manifest = await loadTrackManifest(manifestPath, {
      defaultAllowedChangeScope: ["scripts", "src/bci_autoresearch/models"],
    });
    assert.equal(manifest.reviewCadence, "daily");
    assert.equal(manifest.tracks.length, 2);
    assert.equal(manifest.tracks[0].topicId, "canonical_mainline");
    assert.equal(manifest.tracks[0].runnerFamily, "tree_xgboost");
    assert.equal(manifest.tracks[0].internetResearchEnabled, true);
    assert.deepEqual(manifest.tracks[0].allowedChangeScope, ["scripts", "src/bci_autoresearch/models"]);
    assert.equal(manifest.tracks[1].topicId, "relative_origin_xyz");
    assert.equal(manifest.tracks[1].runnerFamily, "tree_xgboost");
    assert.equal(manifest.tracks[1].internetResearchEnabled, false);
    assert.deepEqual(manifest.tracks[1].allowedChangeScope, ["src/bci_autoresearch/features"]);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("repo plumbing and structure manifests expose multi-family execution plans", async () => {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
  const plumbingPath = path.join(repoRoot, "tools", "autoresearch", "tracks.plumbing.json");
  const structurePath = path.join(repoRoot, "tools", "autoresearch", "tracks.structure.json");
  const currentPath = path.join(repoRoot, "tools", "autoresearch", "tracks.current.json");
  const defaultScope = ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"];

  const plumbing = await loadTrackManifest(plumbingPath, { defaultAllowedChangeScope: defaultScope });
  const structure = await loadTrackManifest(structurePath, { defaultAllowedChangeScope: defaultScope });
  const current = await loadTrackManifest(currentPath, { defaultAllowedChangeScope: defaultScope });

  assert.deepEqual(
    plumbing.tracks.map((track) => track.runnerFamily),
    ["tree_xgboost", "ridge", "feature_lstm"],
  );
  assert.deepEqual(
    structure.tracks.map((track) => track.trackId),
    current.tracks.map((track) => track.trackId),
  );
  assert.ok(current.tracks.some((track) => track.trackId === "canonical_mainline_tree_xgboost"));
  assert.ok(current.tracks.some((track) => track.trackId === "phase_conditioned_feature_lstm"));
  assert.ok(current.tracks.some((track) => track.trackId === "dmd_sdm_ridge"));
});

test("loadTrackManifest assigns default expected memory classes from runner family", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-manifest-memory-"));
  const manifestPath = path.join(tempRoot, "tracks.json");
  await writeFile(
    manifestPath,
    JSON.stringify(
      {
        tracks: [
          {
            track_id: "canonical_mainline_feature_lstm",
            runner_family: "feature_lstm",
            track_goal: "Feature LSTM mainline.",
            promotion_target: "canonical_mainline",
            smoke_command: "python scripts/train_feature_lstm.py",
            formal_command: "python scripts/train_feature_lstm.py --formal",
          },
          {
            track_id: "canonical_mainline_ridge",
            runner_family: "ridge",
            track_goal: "Ridge mainline.",
            promotion_target: "canonical_mainline",
            smoke_command: "python scripts/train_ridge.py",
            formal_command: "python scripts/train_ridge.py --formal",
          },
          {
            track_id: "canonical_mainline_feature_gru",
            runner_family: "feature_gru",
            track_goal: "Feature GRU mainline.",
            promotion_target: "canonical_mainline",
            smoke_command: "python scripts/train_feature_gru.py",
            formal_command: "python scripts/train_feature_gru.py --formal",
          },
          {
            track_id: "canonical_mainline_feature_tcn",
            runner_family: "feature_tcn",
            track_goal: "Feature TCN mainline.",
            promotion_target: "canonical_mainline",
            smoke_command: "python scripts/train_feature_tcn.py",
            formal_command: "python scripts/train_feature_tcn.py --formal",
          },
        ],
      },
      null,
      2,
    ),
    "utf8",
  );

  try {
    const manifest = await loadTrackManifest(manifestPath, {
      defaultAllowedChangeScope: ["scripts"],
    });
    assert.equal(manifest.tracks[0]?.expectedMemoryClass, "high");
    assert.equal(manifest.tracks[1]?.expectedMemoryClass, "low");
    assert.equal(manifest.tracks[2]?.expectedMemoryClass, "high");
    assert.equal(manifest.tracks[3]?.expectedMemoryClass, "high");
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("loadTrackManifest preserves validated skip-edit tracks for preflighted moonshot queues", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-manifest-validated-"));
  const manifestPath = path.join(tempRoot, "tracks.json");
  await writeFile(
    manifestPath,
    JSON.stringify(
      {
        tracks: [
          {
            track_id: "moonshot_upper_bound_feature_gru_lmp_hg_phase_state_scout",
            topic_id: "same_session_pure_brain_moonshot",
            runner_family: "feature_gru",
            track_goal: "Feature GRU moonshot scout.",
            promotion_target: "same_session_pure_brain_moonshot",
            smoke_command: "python scripts/train_feature_gru.py --preflight-only",
            formal_command: "python scripts/train_feature_gru.py --formal",
            validated: true,
          },
        ],
      },
      null,
      2,
    ),
    "utf8",
  );

  try {
    const manifest = await loadTrackManifest(manifestPath, {
      defaultAllowedChangeScope: ["scripts"],
    });
    assert.equal(manifest.tracks[0]?.validated, true);
    assert.equal(manifest.tracks[0]?.skipCodexEdit, true);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("buildCodexPrompt includes the program layers, track context, and output contract", () => {
  const prompt = buildCodexPrompt({
    campaignId: "demo-campaign",
    iteration: 3,
    allowedDirs: ["/repo/scripts", "/repo/src/bci_autoresearch/models"],
    programDocuments: {
      constitutionPath: "/repo/docs/CONSTITUTION.md",
      constitutionText: "# Constitution\nThe repo source of truth lives here.",
      derivedProgramPath: "/repo/tools/autoresearch/program.md",
      derivedProgramText: "# Execution Contract\nNever touch alignment.",
      currentProgramPath: "/repo/tools/autoresearch/program.current.md",
      currentProgramText: "# Current Campaign\nTrack A explores relative-origin targets.",
    },
    track: {
      trackId: "relative_origin_xyz",
      topicId: "relative_origin_xyz",
      runnerFamily: "tree_xgboost",
      internetResearchEnabled: true,
      trackGoal: "Test RSCA-relative xyz abstraction.",
      promotionTarget: "canonical_mainline",
      smokeCommand: "python scripts/train_tree_baseline.py --dataset-config rel_smoke.yaml --model-family xgboost",
      formalCommand: "python scripts/train_tree_baseline.py --dataset-config rel.yaml --model-family xgboost",
      allowedChangeScope: ["src/bci_autoresearch/features"],
    },
    statusSnapshot: {
      stage: "editing",
      acceptedStableBestMetric: 0.4329,
      acceptedStableBestSource: "/repo/artifacts/best.json",
      canonicalPromotionMetric: "val_metrics.mean_pearson_r_zero_lag_macro",
    },
  });

  assert.match(prompt, /CONSTITUTION\.md/);
  assert.match(prompt, /program\.md/);
  assert.match(prompt, /program\.current\.md/);
  assert.match(prompt, /relative_origin_xyz/);
  assert.match(prompt, /topic_id: relative_origin_xyz/);
  assert.match(prompt, /runner_family: tree_xgboost/);
  assert.match(prompt, /internet_research_enabled: true/);
  assert.match(prompt, /Test RSCA-relative xyz abstraction/);
  assert.match(prompt, /change_bucket/);
  assert.match(prompt, /track_comparison_note/);
  assert.match(prompt, /search_queries/);
  assert.match(prompt, /research_evidence/);
  assert.match(prompt, /如果你要联网搜索/);
  assert.match(prompt, /source of truth/);
  assert.match(prompt, /Never touch alignment/);
  assert.match(prompt, /Track A explores relative-origin targets/);
});

test("buildCodexPrompt switches to a minimal closeout prompt when the campaign is in closeout mode", () => {
  const prompt = buildCodexPrompt({
    campaignId: "demo-campaign",
    iteration: 3,
    allowedDirs: ["/repo/scripts"],
    programDocuments: {
      constitutionPath: "/repo/docs/CONSTITUTION.md",
      constitutionText: "# Constitution\nThe repo source of truth lives here.",
      derivedProgramPath: "/repo/tools/autoresearch/program.md",
      derivedProgramText: "# Execution Contract\nNever touch alignment.",
      currentProgramPath: "/repo/tools/autoresearch/program.current.md",
      currentProgramText: "# Current Campaign\nTrack A explores relative-origin targets.",
    },
    track: {
      trackId: "canonical_mainline_feature_lstm",
      topicId: "canonical_mainline",
      runnerFamily: "feature_lstm",
      internetResearchEnabled: false,
      trackGoal: "Protect the canonical joints mainline.",
      promotionTarget: "canonical_mainline",
      smokeCommand: "python scripts/train_feature_lstm.py --dataset-config joints_smoke.yaml",
      formalCommand: "python scripts/train_feature_lstm.py --dataset-config joints.yaml",
      allowedChangeScope: ["scripts"],
    },
    statusSnapshot: {
      stage: "smoke",
      acceptedStableBestMetric: 0.4329,
      acceptedStableBestSource: "/repo/artifacts/best.json",
      canonicalPromotionMetric: "val_metrics.mean_pearson_r_zero_lag_macro",
    },
    promptMode: "closeout",
  });

  assert.match(prompt, /收尾模式/);
  assert.match(prompt, /最近一次结果/);
  assert.doesNotMatch(prompt, /CONSTITUTION\.md/);
  assert.doesNotMatch(prompt, /program\.current\.md/);
  assert.doesNotMatch(prompt, /如果你要联网搜索/);
});

test("buildDailyReviewPacket summarizes tracks and promotion candidates for one day", () => {
  const records: CampaignRecordLike[] = [
    {
      campaign_id: "demo-campaign",
      run_id: "run-001",
      recorded_at: "2026-04-07T08:00:00.000Z",
      track_id: "canonical_mainline",
      track_goal: "Protect the mainline.",
      decision: "smoke_not_better",
      change_bucket: "model-led",
      track_comparison_note: "Did not beat the stable best.",
      smoke_metrics: { val_primary_metric: 0.39, test_primary_metric: 0.33 },
      final_metrics: null,
    },
    {
      campaign_id: "demo-campaign",
      run_id: "run-002",
      recorded_at: "2026-04-07T10:30:00.000Z",
      track_id: "relative_origin_xyz",
      track_goal: "Explore RSCA-relative xyz.",
      decision: "hold_for_promotion_review",
      change_bucket: "representation-led",
      track_comparison_note: "Local score improved; canonical retest required.",
      smoke_metrics: { val_primary_metric: 0.46, test_primary_metric: 0.41 },
      final_metrics: { val_primary_metric: 0.47, test_primary_metric: 0.42 },
    },
  ];

  const packet = buildDailyReviewPacket({
    campaignId: "demo-campaign",
    reviewDate: "2026-04-07",
    records,
  });

  assert.equal(packet.reviewDate, "2026-04-07");
  assert.equal(packet.trackSummaries.length, 2);
  assert.equal(packet.trackSummaries[1]?.trackId, "relative_origin_xyz");
  assert.equal(packet.promotionCandidates.length, 1);
  assert.equal(packet.promotionCandidates[0]?.trackId, "relative_origin_xyz");
});

test("decideTrackOutcome separates local track wins from canonical promotion wins", () => {
  const exploratory = decideTrackOutcome({
    promotionTarget: "canonical_mainline",
    formalValMetric: 0.47,
    localBestMetric: 0.41,
    globalStableBestMetric: 0.4329,
    canonicalRetestReady: false,
  });
  assert.equal(exploratory.decision, "hold_for_promotion_review");
  assert.equal(exploratory.updateLocalBest, true);
  assert.equal(exploratory.updateGlobalCandidate, false);

  const canonical = decideTrackOutcome({
    promotionTarget: "canonical_mainline",
    formalValMetric: 0.48,
    localBestMetric: 0.45,
    globalStableBestMetric: 0.4329,
    canonicalRetestReady: true,
  });
  assert.equal(canonical.decision, "hold_for_packet_gate");
  assert.equal(canonical.updateLocalBest, true);
  assert.equal(canonical.updateGlobalCandidate, true);
});

test("renderDailyReviewPacketMarkdown includes track summaries and promotion candidates", () => {
  const markdown = renderDailyReviewPacketMarkdown({
    campaignId: "demo-campaign",
    reviewDate: "2026-04-07",
    generatedAt: "2026-04-07T12:00:00.000Z",
    trackSummaries: [
      {
        trackId: "canonical_mainline",
        trackGoal: "Protect the mainline.",
        runCount: 2,
        latestDecision: "smoke_not_better",
        latestRunId: "run-001",
        latestChangeBucket: "model-led",
        latestComparisonNote: "No global improvement.",
        bestValMetric: 0.41,
        bestTestMetric: 0.33,
      },
    ],
    promotionCandidates: [
      {
        trackId: "relative_origin_xyz",
        runId: "run-002",
        decision: "hold_for_promotion_review",
        comparisonNote: "Canonical retest required.",
        valMetric: 0.47,
        testMetric: 0.42,
      },
    ],
  });

  assert.match(markdown, /demo-campaign/);
  assert.match(markdown, /canonical_mainline/);
  assert.match(markdown, /relative_origin_xyz/);
  assert.match(markdown, /Canonical retest required/);
});
