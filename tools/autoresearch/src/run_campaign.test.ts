import assert from "node:assert/strict";
import test from "node:test";

import {
  applyTrackRuntimeResultSummary,
  buildResearchLogEntries,
  canTrackStartNewEditTurn,
  campaignReasoningEffortForMode,
  classifyCommandRelevance,
  deriveCampaignControlState,
  hydrateTrackRuntimeState,
  parseProposal,
  shouldForceFreshThread,
  summarizeThreadItemUsage,
  deriveTurnTelemetry,
  shouldTrackSkipCodexEdit,
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
    best_checkpoint_path: null,
    last_checkpoint_path: null,
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
    track_origin: "default",
    force_fresh_thread: false,
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
    method_variant_label: "",
    input_mode_label: "",
    series_class: "",
    promotable: true,
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
  } as any;
}

function track(trackId: string, runnerFamily = "xgboost") {
  return {
    trackId,
    trackGoal: "demo goal",
    promotionTarget: "canonical_mainline",
    smokeCommand: "python smoke.py",
    formalCommand: "python formal.py",
    allowedChangeScope: ["scripts"],
    runnerFamily,
    validated: false,
    skipCodexEdit: false,
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

test("classifyCommandRelevance treats feature_gru and feature_tcn runners like feature sequence tracks", () => {
  const gruRelevance = classifyCommandRelevance(
    ["scripts/train_feature_gru.py", "src/bci_autoresearch/models/gru_regressor.py"],
    ".venv/bin/python scripts/train_feature_gru.py --dataset-config joints.yaml --final-eval",
    ".venv/bin/python scripts/train_feature_gru.py --dataset-config joints.yaml --final-eval",
  );
  assert.equal(gruRelevance.label, "on_track");
  assert.equal(gruRelevance.hardBlock, false);

  const tcnRelevance = classifyCommandRelevance(
    ["scripts/train_feature_tcn.py", "src/bci_autoresearch/models/tcn_regressor.py"],
    ".venv/bin/python scripts/train_feature_tcn.py --dataset-config joints.yaml --final-eval",
    ".venv/bin/python scripts/train_feature_tcn.py --dataset-config joints.yaml --final-eval",
  );
  assert.equal(tcnRelevance.label, "on_track");
  assert.equal(tcnRelevance.hardBlock, false);
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

test("campaignReasoningEffortForMode lowers reasoning for closeout while keeping exploration at medium", () => {
  assert.equal(campaignReasoningEffortForMode("exploration"), "medium");
  assert.equal(campaignReasoningEffortForMode("closeout"), "low");
});

test("shouldTrackSkipCodexEdit only bypasses codex edit for validated preflighted tracks", () => {
  assert.equal(
    shouldTrackSkipCodexEdit({
      ...track("moonshot_upper_bound_feature_gru_lmp_hg_phase_state_scout", "feature_gru"),
      validated: true,
      skipCodexEdit: true,
    }),
    true,
  );

  assert.equal(
    shouldTrackSkipCodexEdit({
      ...track("canonical_mainline_feature_gru", "feature_gru"),
      validated: false,
      skipCodexEdit: false,
    }),
    false,
  );
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

test("applyTrackRuntimeResultSummary records latest formal metrics and human-readable labels", () => {
  const state = trackRuntimeState("phase_conditioned_feature_lstm");

  applyTrackRuntimeResultSummary({
    trackState: state,
    track: track("phase_conditioned_feature_lstm", "feature_lstm"),
    runId: "phase-conditioned-feature-lstm-001",
    smokeMetrics: {
      ...acceptedSnapshot(),
      source_path: "smoke.json",
      result_json: "smoke.json",
      val_primary_metric: 0.2327,
      formal_val_primary_metric: 0.2327,
      val_rmse: 11.4938,
      test_primary_metric: 0.2996,
      test_rmse: 11.1166,
      feature_family: "lmp+hg_power",
      model_family: "feature_lstm",
      experiment_track: null,
      evaluation_mode: "cross_session_mainline",
    },
    finalMetrics: {
      ...acceptedSnapshot(),
      source_path: "formal.json",
      result_json: "formal.json",
      val_primary_metric: 0.4413,
      formal_val_primary_metric: 0.4413,
      val_rmse: 10.1983,
      test_primary_metric: 0.3902,
      test_rmse: 11.4573,
      feature_family: "lmp+hg_power",
      model_family: "feature_lstm",
      experiment_track: null,
      evaluation_mode: "cross_session_mainline",
    },
    decision: "hold_for_packet_gate",
  });

  assert.equal(state.latest_run_id, "phase-conditioned-feature-lstm-001");
  assert.equal(state.latest_smoke_run_id, "phase-conditioned-feature-lstm-001");
  assert.equal(state.latest_formal_run_id, "phase-conditioned-feature-lstm-001");
  assert.equal(state.latest_val_primary_metric, 0.4413);
  assert.equal(state.latest_test_primary_metric, 0.3902);
  assert.equal(state.latest_val_rmse, 10.1983);
  assert.equal(state.best_val_primary_metric, 0.4413);
  assert.equal(state.best_test_primary_metric, 0.3902);
  assert.equal(state.method_variant_label, "phase 条件版");
  assert.equal(state.input_mode_label, "只用脑电");
  assert.equal(state.series_class, "mainline_brain");
  assert.equal(state.promotable, true);
  assert.match(state.last_result_summary, /已正式比较/);
});

test("applyTrackRuntimeResultSummary humanizes feature_gru and feature_tcn families", () => {
  const gruState = trackRuntimeState("canonical_mainline_feature_gru");
  applyTrackRuntimeResultSummary({
    trackState: gruState,
    track: track("canonical_mainline_feature_gru", "feature_gru"),
    runId: "feature-gru-001",
    smokeMetrics: {
      ...acceptedSnapshot(),
      source_path: "smoke.json",
      result_json: "smoke.json",
      val_primary_metric: 0.3012,
      formal_val_primary_metric: 0.3012,
      val_rmse: 10.8,
      test_primary_metric: 0.2555,
      test_rmse: 11.2,
      feature_family: "lmp+hg_power",
      model_family: "feature_gru",
      experiment_track: null,
      evaluation_mode: "cross_session_mainline",
    },
    finalMetrics: null,
    decision: "hold_for_promotion_review",
  });
  assert.match(gruState.last_result_summary, /Feature GRU/);

  const tcnState = trackRuntimeState("canonical_mainline_feature_tcn");
  applyTrackRuntimeResultSummary({
    trackState: tcnState,
    track: track("canonical_mainline_feature_tcn", "feature_tcn"),
    runId: "feature-tcn-001",
    smokeMetrics: {
      ...acceptedSnapshot(),
      source_path: "smoke.json",
      result_json: "smoke.json",
      val_primary_metric: 0.2876,
      formal_val_primary_metric: 0.2876,
      val_rmse: 10.4,
      test_primary_metric: 0.2491,
      test_rmse: 11.0,
      feature_family: "lmp+hg_power",
      model_family: "feature_tcn",
      experiment_track: null,
      evaluation_mode: "cross_session_mainline",
    },
    finalMetrics: null,
    decision: "hold_for_promotion_review",
  });
  assert.match(tcnState.last_result_summary, /Feature TCN/);
});

test("applyTrackRuntimeResultSummary keeps rollback failures visible for wave1 control and feature tracks", () => {
  const state = trackRuntimeState("hybrid_brain_plus_kinematics");

  applyTrackRuntimeResultSummary({
    trackState: state,
    track: track("hybrid_brain_plus_kinematics", "feature_lstm"),
    runId: "hybrid-001",
    smokeMetrics: null,
    finalMetrics: null,
    decision: "rollback_command_failed",
  });

  assert.equal(state.latest_run_id, "hybrid-001");
  assert.equal(state.latest_val_primary_metric, null);
  assert.equal(state.method_variant_label, "混合输入（脑电 + 运动学历史）");
  assert.equal(state.input_mode_label, "脑电 + 运动学历史");
  assert.equal(state.series_class, "control");
  assert.equal(state.promotable, false);
  assert.match(state.last_result_summary, /回滚\/命令失败/);
});

test("hydrateTrackRuntimeState backfills legacy wave1 labels and promotable from track id", () => {
  const hydrated = hydrateTrackRuntimeState(
    {
      ...trackRuntimeState("relative_origin_xyz"),
      method_variant_label: "",
      input_mode_label: "",
      series_class: "",
      promotable: false,
    },
    track("relative_origin_xyz", "feature_lstm"),
  );

  assert.equal(hydrated.method_variant_label, "相对 RSCA 三方向坐标");
  assert.equal(hydrated.input_mode_label, "只用脑电");
  assert.equal(hydrated.series_class, "structure");
  assert.equal(hydrated.promotable, false);
});

test("hydrateTrackRuntimeState preserves incubation thread metadata from the track", () => {
  const hydrated = hydrateTrackRuntimeState(
    {
      ...trackRuntimeState("incubation_feature_gru"),
      track_origin: "default",
      force_fresh_thread: false,
    },
    {
      ...track("incubation_feature_gru", "feature_gru"),
      trackOrigin: "incubation",
      forceFreshThread: true,
    } as any,
  );

  assert.equal(hydrated.track_origin, "incubation");
  assert.equal(hydrated.force_fresh_thread, true);
});

test("shouldForceFreshThread treats incubation tracks as fresh-thread only", () => {
  assert.equal(shouldForceFreshThread({ trackOrigin: "incubation", forceFreshThread: false } as any), true);
  assert.equal(shouldForceFreshThread({ trackOrigin: "default", forceFreshThread: true } as any), true);
  assert.equal(shouldForceFreshThread({ trackOrigin: "default", forceFreshThread: false } as any), false);
});

test("summarizeThreadItemUsage counts the major Codex turn item types", () => {
  const summary = summarizeThreadItemUsage([
    { id: "r1", type: "reasoning", text: "thinking" },
    { id: "m1", type: "agent_message", text: "hello" },
    {
      id: "c1",
      type: "command_execution",
      command: "python -m pytest",
      aggregated_output: "",
      exit_code: 0,
      status: "completed",
    },
    {
      id: "f1",
      type: "file_change",
      changes: [{ path: "src/demo.ts", kind: "update" }],
      status: "completed",
    },
    {
      id: "w1",
      type: "web_search",
      query: "new bci paper",
    },
    {
      id: "t1",
      type: "todo_list",
      items: [{ text: "ship", completed: false }],
    },
    {
      id: "e1",
      type: "error",
      message: "oops",
    },
    {
      id: "mcp1",
      type: "mcp_tool_call",
      server: "github",
      tool: "search",
      arguments: {},
      status: "completed",
    },
  ] as any);

  assert.deepEqual(summary, {
    total_items: 8,
    reasoning_items: 1,
    agent_messages: 1,
    command_executions: 1,
    file_changes: 1,
    web_searches: 1,
    mcp_tool_calls: 1,
    todo_lists: 1,
    errors: 1,
    completed_items: 4,
    failed_items: 0,
  });
});

test("deriveTurnTelemetry surfaces stale and pivot signals from the turn items", () => {
  const telemetry = deriveTurnTelemetry({
    existing: {
      thinking_heartbeat_at: "",
      last_retrieval_at: "",
      last_decision_at: "",
      last_judgment_at: "",
      last_materialization_at: "",
      last_smoke_at: "",
      stale_reason_codes: [],
      pivot_reason_codes: [],
      search_budget_state: "healthy",
      tool_usage_summary: null,
    },
    items: [
      {
        id: "w1",
        type: "web_search",
        query: "new paper",
      },
      {
        id: "c1",
        type: "command_execution",
        command: "python smoke.py",
        aggregated_output: "",
        exit_code: 0,
        status: "completed",
      },
    ] as any,
    decision: "smoke_not_better",
    smokeMetrics: null,
    finalMetrics: null,
    nowIso: "2026-04-12T00:00:00.000Z",
  });

  assert.equal(telemetry.thinking_heartbeat_at, "2026-04-12T00:00:00.000Z");
  assert.equal(telemetry.last_retrieval_at, "2026-04-12T00:00:00.000Z");
  assert.equal(telemetry.last_smoke_at, "2026-04-12T00:00:00.000Z");
  assert.deepEqual(telemetry.stale_reason_codes, ["no_new_smoke"]);
  assert.deepEqual(telemetry.pivot_reason_codes, ["needs_new_direction"]);
  assert.equal(telemetry.tool_usage_summary?.web_searches, 1);
});

test("applyTrackRuntimeResultSummary keeps latest and best fields stable across accepted and rollback results", () => {
  const acceptedState = trackRuntimeState("phase_conditioned_feature_lstm");
  applyTrackRuntimeResultSummary({
    trackState: acceptedState,
    track: track("phase_conditioned_feature_lstm", "feature_lstm"),
    runId: "phase-conditioned-feature-lstm-accepted",
    smokeMetrics: {
      ...acceptedSnapshot(),
      source_path: "smoke.json",
      result_json: "smoke.json",
      val_primary_metric: 0.321,
      formal_val_primary_metric: 0.321,
      val_rmse: 12.3,
      test_primary_metric: 0.29,
      test_rmse: 12.8,
      feature_family: "lmp+hg_power",
      model_family: "feature_lstm",
      experiment_track: null,
      evaluation_mode: "cross_session_mainline",
    },
    finalMetrics: {
      ...acceptedSnapshot(),
      source_path: "formal.json",
      result_json: "formal.json",
      val_primary_metric: 0.4413,
      formal_val_primary_metric: 0.4413,
      val_rmse: 10.1983,
      test_primary_metric: 0.3902,
      test_rmse: 11.4573,
      feature_family: "lmp+hg_power",
      model_family: "feature_lstm",
      experiment_track: null,
      evaluation_mode: "cross_session_mainline",
    },
    decision: "accepted",
  });

  assert.equal(acceptedState.latest_run_id, "phase-conditioned-feature-lstm-accepted");
  assert.equal(acceptedState.latest_formal_run_id, "phase-conditioned-feature-lstm-accepted");
  assert.equal(acceptedState.latest_val_primary_metric, 0.4413);
  assert.equal(acceptedState.best_val_primary_metric, 0.4413);
  assert.equal(acceptedState.method_variant_label, "phase 条件版");
  assert.equal(acceptedState.input_mode_label, "只用脑电");
  assert.equal(acceptedState.series_class, "mainline_brain");
  assert.equal(acceptedState.promotable, true);
  assert.match(acceptedState.last_result_summary, /已正式比较/);

  const rollbackState = trackRuntimeState("hybrid_brain_plus_kinematics") as any;
  rollbackState.best_val_primary_metric = 0.48;
  rollbackState.best_test_primary_metric = 0.43;
  rollbackState.best_val_rmse = 9.5;
  rollbackState.best_test_rmse = 10.1;

  applyTrackRuntimeResultSummary({
    trackState: rollbackState,
    track: track("hybrid_brain_plus_kinematics", "feature_lstm"),
    runId: "hybrid-rollback",
    smokeMetrics: null,
    finalMetrics: null,
    decision: "rollback",
  });

  assert.equal(rollbackState.latest_run_id, "hybrid-rollback");
  assert.equal(rollbackState.latest_val_primary_metric, null);
  assert.equal(rollbackState.best_val_primary_metric, 0.48);
  assert.equal(rollbackState.best_test_primary_metric, 0.43);
  assert.equal(rollbackState.method_variant_label, "混合输入（脑电 + 运动学历史）");
  assert.equal(rollbackState.input_mode_label, "脑电 + 运动学历史");
  assert.equal(rollbackState.series_class, "control");
  assert.equal(rollbackState.promotable, false);
  assert.match(rollbackState.last_result_summary, /rollback/);
});
