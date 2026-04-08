import assert from "node:assert/strict";
import test from "node:test";

import {
  buildResearchLogEntries,
  canTrackStartNewEditTurn,
  classifyCommandRelevance,
  deriveCampaignControlState,
  parseProposal,
} from "./run_campaign.js";

function acceptedSnapshot(runId = "") {
  return {
    run_id: runId,
    dataset_name: "walk_matched_v1_64clean_joints",
    target_mode: "joints_sheet",
    target_space: "joint_angle",
    primary_metric_name: "val_metrics.mean_pearson_r_zero_lag_macro",
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

function trackRuntimeState(trackId: string) {
  return {
    track_id: trackId,
    track_goal: "demo goal",
    promotion_target: "canonical_mainline",
    smoke_command: "python smoke.py",
    formal_command: "python formal.py",
    allowed_change_scope: ["scripts"],
    current_iteration: 1,
    patience_streak: 0,
    edit_turns_used: 1,
    formal_runs_completed: 0,
    stage: "editing" as const,
    last_run_id: "",
    last_decision: "",
    codex_thread_id: null,
    local_best: acceptedSnapshot(),
    updated_at: "2026-04-08T00:00:00.000Z",
  };
}

test("classifyCommandRelevance treats test files as supporting changes instead of a hard block", () => {
  const relevance = classifyCommandRelevance(
    ["scripts/train_tree_baseline.py", "tests/test_train_tree_baseline.py"],
    ".venv/bin/python scripts/train_tree_baseline.py --model-family xgboost",
    ".venv/bin/python scripts/train_tree_baseline.py --model-family xgboost",
  );

  assert.equal(relevance.label, "on_track");
  assert.equal(relevance.hardBlock, false);
  assert.match(relevance.reason, /核心改动/);
});

test("classifyCommandRelevance keeps indirect plumbing changes runnable and only labels them after the fact", () => {
  const relevance = classifyCommandRelevance(
    ["scripts/train_feature_lstm.py"],
    ".venv/bin/python scripts/train_ridge.py --ridge-alpha 1.0",
    ".venv/bin/python scripts/train_ridge.py --ridge-alpha 1.0",
  );

  assert.equal(relevance.label, "exploratory_but_indirect");
  assert.equal(relevance.hardBlock, false);
});

test("parseProposal captures external search queries and evidence when present", () => {
  const proposal = parseProposal(
    JSON.stringify({
      hypothesis: "Try a more stable relative target.",
      why_this_change: "Recent literature suggests relative targets reduce drift.",
      changes_summary: "Adjusted target handling and searched for relative-coordinate BCI references.",
      change_bucket: "representation-led",
      track_comparison_note: "Still needs canonical retest.",
      files_touched: ["scripts/train_feature_lstm.py"],
      next_step: "Run smoke.",
      search_queries: [
        {
          search_query: "relative coordinate BCI decoding xgboost",
          search_intent: "paper",
        },
      ],
      research_evidence: [
        {
          search_query: "relative coordinate BCI decoding xgboost",
          search_intent: "paper",
          source_type: "paper",
          source_url: "https://example.com/paper",
          source_title: "Relative Coordinate Decoding",
          why_it_matters: "Supports trying relative targets before stricter regularization.",
        },
      ],
    }),
  );

  assert.equal(proposal.search_queries.length, 1);
  assert.equal(proposal.research_evidence.length, 1);
  assert.equal(proposal.search_queries[0]?.search_query, "relative coordinate BCI decoding xgboost");
  assert.equal(proposal.research_evidence[0]?.source_type, "paper");
});

test("buildResearchLogEntries writes one query row and one evidence row per cited source", () => {
  const entries = buildResearchLogEntries({
    campaignId: "overnight-2026-04-08-struct-r05",
    runId: "overnight-2026-04-08-struct-r05-relative_origin_xyz_feature_lstm-iter-001",
    trackId: "relative_origin_xyz_feature_lstm",
    recordedAt: "2026-04-08T12:00:00.000Z",
    searchQueries: [
      {
        search_query: "feature lstm bci relative marker targets",
        search_intent: "engineering",
      },
    ],
    researchEvidence: [
      {
        search_query: "feature lstm bci relative marker targets",
        search_intent: "engineering",
        source_type: "github_repo",
        source_url: "https://github.com/example/repo",
        source_title: "Feature LSTM decoder",
        why_it_matters: "Shows one way to combine pooled hidden states.",
      },
    ],
  });

  assert.equal(entries.queryEntries.length, 1);
  assert.equal(entries.evidenceEntries.length, 1);
  assert.equal(entries.queryEntries[0]?.used_in_run_id, "overnight-2026-04-08-struct-r05-relative_origin_xyz_feature_lstm-iter-001");
  assert.equal(entries.evidenceEntries[0]?.source_title, "Feature LSTM decoder");
});

test("deriveCampaignControlState enters closeout when a leading candidate appears", () => {
  const control = deriveCampaignControlState({
    status: {
      campaign_id: "overnight-2026-04-08-struct-r06",
      stage: "smoke",
      started_at: "2026-04-08T00:00:00.000Z",
      stop_reason: "none",
      leading_unverified_candidate: acceptedSnapshot("candidate-001"),
      track_states: [trackRuntimeState("canonical_mainline_feature_lstm")],
    },
    nowIso: "2026-04-08T01:00:00.000Z",
  });

  assert.equal(control.campaignMode, "closeout");
  assert.equal(control.budgetState, "healthy");
  assert.equal(control.stopReason, "none");
});

test("deriveCampaignControlState caps budget and marks wall clock stop when the campaign exceeds four hours", () => {
  const control = deriveCampaignControlState({
    status: {
      campaign_id: "overnight-2026-04-08-struct-r06",
      stage: "smoke",
      started_at: "2026-04-08T00:00:00.000Z",
      stop_reason: "none",
      leading_unverified_candidate: acceptedSnapshot(""),
      track_states: [trackRuntimeState("canonical_mainline_feature_lstm")],
    },
    nowIso: "2026-04-08T04:00:01.000Z",
  });

  assert.equal(control.campaignMode, "closeout");
  assert.equal(control.budgetState, "capped");
  assert.equal(control.stopReason, "wall_clock_cap");
});

test("canTrackStartNewEditTurn stops tracks after two edit turns or two no-improvement rounds", () => {
  assert.equal(
    canTrackStartNewEditTurn(
      {
        current_iteration: 1,
        patience_streak: 2,
        edit_turns_used: 1,
      },
      {
        maxIterations: 8,
        consecutiveLimit: 3,
        maxEditTurnsPerTrack: 2,
        noImprovementLimit: 2,
      },
    ),
    false,
  );

  assert.equal(
    canTrackStartNewEditTurn(
      {
        current_iteration: 2,
        patience_streak: 0,
        edit_turns_used: 2,
      },
      {
        maxIterations: 8,
        consecutiveLimit: 3,
        maxEditTurnsPerTrack: 2,
        noImprovementLimit: 2,
      },
    ),
    false,
  );

  assert.equal(
    canTrackStartNewEditTurn(
      {
        current_iteration: 1,
        patience_streak: 1,
        edit_turns_used: 1,
      },
      {
        maxIterations: 8,
        consecutiveLimit: 3,
        maxEditTurnsPerTrack: 2,
        noImprovementLimit: 2,
      },
    ),
    true,
  );
});
