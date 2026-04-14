import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  applyRuntimeCampaignOverlay,
  loadRuntimeCampaignOverlay,
  parseRetryCampaignId,
  resolveRuntimeOverlayCandidates,
} from "./runtime_campaign.js";

test("parseRetryCampaignId splits retry suffixes from overnight campaign ids", () => {
  assert.deepEqual(parseRetryCampaignId("overnight-2026-04-07-struct-r01"), {
    baseCampaignId: "overnight-2026-04-07-struct",
    retrySuffix: "r01",
  });
  assert.deepEqual(parseRetryCampaignId("overnight-2026-04-07-struct"), {
    baseCampaignId: "overnight-2026-04-07-struct",
    retrySuffix: null,
  });
});

test("resolveRuntimeOverlayCandidates prefers retry-specific overlay paths", () => {
  const candidates = resolveRuntimeOverlayCandidates("/repo", "overnight-2026-04-07-struct-r02");
  assert.deepEqual(candidates, [
    "/repo/artifacts/monitor/runtime_overrides/overnight-2026-04-07-struct-r02.json",
    "/repo/artifacts/monitor/runtime_overrides/overnight-2026-04-07-struct/r02.json",
    "/repo/artifacts/monitor/runtime_overrides/overnight-2026-04-07-struct.json",
  ]);
});

test("loadRuntimeCampaignOverlay reads the default retry overlay file when it exists", async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "autoresearch-overlay-"));
  const overlayPath = path.join(
    tempRoot,
    "artifacts",
    "monitor",
    "runtime_overrides",
    "overnight-2026-04-07-struct",
    "r01.json",
  );
  await mkdir(path.dirname(overlayPath), { recursive: true });
  await writeFile(
    overlayPath,
    JSON.stringify(
      {
        skip_track_ids: ["relative_origin_xyz_upper_bound"],
        append_tracks: [
          {
            track_id: "incubation_feature_gru",
            track_goal: "Try a new incubation direction.",
            promotion_target: "canonical_mainline",
            smoke_command: "python smoke_incubation.py",
            formal_command: "python formal_incubation.py",
            allowed_change_scope: ["scripts"],
            internet_research_enabled: true,
            track_origin: "incubation",
            force_fresh_thread: true,
          },
        ],
      },
      null,
      2,
    ),
    "utf8",
  );

  try {
    const overlay = await loadRuntimeCampaignOverlay({
      repoRoot: tempRoot,
      campaignId: "overnight-2026-04-07-struct-r01",
    });

    assert.equal(overlay?.sourcePath, overlayPath);
    assert.deepEqual(overlay?.overlay.skip_track_ids, ["relative_origin_xyz_upper_bound"]);
    assert.equal(overlay?.overlay.append_tracks?.[0]?.track_id, "incubation_feature_gru");
    assert.equal(overlay?.overlay.append_tracks?.[0]?.track_origin, "incubation");
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});

test("applyRuntimeCampaignOverlay skips quarantined tracks and applies runtime-only overrides", () => {
  const tracks = [
    {
      trackId: "canonical_mainline",
      trackGoal: "Protect the canonical joints mainline.",
      promotionTarget: "canonical_mainline",
      smokeCommand: "smoke-a",
      formalCommand: "formal-a",
      allowedChangeScope: ["scripts"],
    },
    {
      trackId: "relative_origin_xyz",
      trackGoal: "Explore RSCA-relative xyz targets.",
      promotionTarget: "canonical_mainline",
      smokeCommand: "smoke-b",
      formalCommand: "formal-b",
      allowedChangeScope: ["src/bci_autoresearch/features"],
    },
  ];

  const applied = applyRuntimeCampaignOverlay(tracks, {
    sourcePath: "/repo/artifacts/monitor/runtime_overrides/demo.json",
    overlay: {
      skip_track_ids: ["relative_origin_xyz"],
      tracks: [
        {
          track_id: "canonical_mainline",
          track_goal: "Runtime overlay goal",
          smoke_command: "smoke-runtime",
          formal_command: "formal-runtime",
        },
      ],
    },
  });

  assert.deepEqual(applied.skippedTrackIds, ["relative_origin_xyz"]);
  assert.equal(applied.tracks.length, 1);
  assert.equal(applied.tracks[0]?.trackGoal, "Runtime overlay goal");
  assert.equal(applied.tracks[0]?.smokeCommand, "smoke-runtime");
  assert.equal(applied.tracks[0]?.formalCommand, "formal-runtime");
  assert.deepEqual(applied.tracks[0]?.allowedChangeScope, ["scripts"]);
  assert.equal(tracks[0].trackGoal, "Protect the canonical joints mainline.");
  assert.equal(tracks[1].trackGoal, "Explore RSCA-relative xyz targets.");
});

test("applyRuntimeCampaignOverlay appends incubation tracks with fresh-thread metadata", () => {
  const tracks = [
    {
      trackId: "canonical_mainline",
      trackGoal: "Protect the canonical joints mainline.",
      promotionTarget: "canonical_mainline",
      smokeCommand: "smoke-a",
      formalCommand: "formal-a",
      allowedChangeScope: ["scripts"],
    },
  ];

  const applied = applyRuntimeCampaignOverlay(tracks, {
    sourcePath: "/repo/artifacts/monitor/runtime_overrides/incubation.json",
    overlay: {
      append_tracks: [
        {
          track_id: "incubation_feature_tcn",
          track_goal: "Probe a new direction.",
          promotion_target: "canonical_mainline",
          smoke_command: "smoke-incubation",
          formal_command: "formal-incubation",
          allowed_change_scope: ["scripts"],
          internet_research_enabled: true,
          track_origin: "incubation",
          force_fresh_thread: true,
        },
      ],
    },
  });

  assert.equal(applied.tracks.length, 2);
  assert.equal(applied.tracks[1]?.trackId, "incubation_feature_tcn");
  assert.equal((applied.tracks[1] as any)?.trackOrigin, "incubation");
  assert.equal((applied.tracks[1] as any)?.forceFreshThread, true);
  assert.equal(tracks.length, 1);
});

test("applyRuntimeCampaignOverlay rejects append tracks that collide with existing ids", () => {
  assert.throws(() => {
    applyRuntimeCampaignOverlay(
      [
        {
          trackId: "canonical_mainline",
          trackGoal: "Protect the canonical joints mainline.",
          promotionTarget: "canonical_mainline",
          smokeCommand: "smoke-a",
          formalCommand: "formal-a",
          allowedChangeScope: ["scripts"],
        },
      ],
      {
        sourcePath: "/repo/artifacts/monitor/runtime_overrides/incubation.json",
        overlay: {
          append_tracks: [
            {
              track_id: "canonical_mainline",
              track_goal: "Duplicate should fail.",
              promotion_target: "canonical_mainline",
              smoke_command: "smoke-incubation",
              formal_command: "formal-incubation",
              allowed_change_scope: ["scripts"],
            },
          ],
        },
      },
    );
  }, /append_tracks/i);
});
