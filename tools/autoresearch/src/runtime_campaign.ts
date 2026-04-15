import { access, readFile } from "node:fs/promises";
import path from "node:path";

import type { CampaignTrack } from "./campaign_program.js";

export type RuntimeTrackOrigin = "default" | "incubation";

export interface RuntimeTrackOverride {
  track_id: string;
  enabled?: boolean;
  track_goal?: string;
  promotion_target?: string;
  smoke_command?: string;
  formal_command?: string;
  allowed_change_scope?: string[];
  internet_research_enabled?: boolean;
  track_origin?: RuntimeTrackOrigin;
  force_fresh_thread?: boolean;
}

export interface RuntimeCampaignOverlay {
  skip_track_ids?: string[];
  quarantined_track_ids?: string[];
  tracks?: RuntimeTrackOverride[];
  append_tracks?: RuntimeTrackOverride[];
}

export interface ResolvedRuntimeCampaignOverlay {
  sourcePath: string;
  overlay: RuntimeCampaignOverlay;
}

export interface RuntimeCampaignTrack extends CampaignTrack {
  trackOrigin?: RuntimeTrackOrigin;
  forceFreshThread?: boolean;
}

export interface AppliedRuntimeCampaignOverlay {
  tracks: RuntimeCampaignTrack[];
  skippedTrackIds: string[];
}

export function parseRetryCampaignId(campaignId: string): { baseCampaignId: string; retrySuffix: string | null } {
  const match = campaignId.match(/^(.*)-r(\d{2,})$/);
  if (!match) {
    return { baseCampaignId: campaignId, retrySuffix: null };
  }

  return {
    baseCampaignId: match[1],
    retrySuffix: `r${match[2]}`,
  };
}

export function resolveRuntimeOverlayCandidates(repoRoot: string, campaignId: string): string[] {
  const overlayRoot = path.join(repoRoot, "artifacts", "monitor", "runtime_overrides");
  const { baseCampaignId, retrySuffix } = parseRetryCampaignId(campaignId);

  const candidates = [path.join(overlayRoot, `${campaignId}.json`)];
  if (retrySuffix) {
    candidates.push(path.join(overlayRoot, baseCampaignId, `${retrySuffix}.json`));
    candidates.push(path.join(overlayRoot, `${baseCampaignId}.json`));
  }

  return dedupeStrings(candidates);
}

export async function loadRuntimeCampaignOverlay({
  repoRoot,
  campaignId,
  overlayInput,
}: {
  repoRoot: string;
  campaignId: string;
  overlayInput?: string | null;
}): Promise<ResolvedRuntimeCampaignOverlay | null> {
  const trimmed = overlayInput?.trim();
  if (trimmed) {
    if (looksLikeInlineJson(trimmed)) {
      return {
        sourcePath: "<inline>",
        overlay: parseRuntimeCampaignOverlay(trimmed),
      };
    }

    const sourcePath = resolveRepoPath(repoRoot, trimmed);
    const text = await readFile(sourcePath, "utf8");
    return {
      sourcePath,
      overlay: parseRuntimeCampaignOverlay(text),
    };
  }

  for (const candidate of resolveRuntimeOverlayCandidates(repoRoot, campaignId)) {
    if (!(await pathExists(candidate))) {
      continue;
    }

    const text = await readFile(candidate, "utf8");
    return {
      sourcePath: candidate,
      overlay: parseRuntimeCampaignOverlay(text),
    };
  }

  return null;
}

export function applyRuntimeCampaignOverlay(
  tracks: CampaignTrack[],
  overlay: ResolvedRuntimeCampaignOverlay | null,
): AppliedRuntimeCampaignOverlay {
  if (!overlay) {
    return {
      tracks: tracks.map((track) => cloneTrack(track)),
      skippedTrackIds: [],
    };
  }

  const normalizedTracks = overlay.overlay.tracks ?? [];
  const normalizedAppendTracks = overlay.overlay.append_tracks ?? [];
  const normalizedById = new Map(normalizedTracks.map((item) => [item.track_id, item]));
  const appendById = new Map<string, RuntimeTrackOverride>();
  const skipTrackIds = dedupeStrings([
    ...(overlay.overlay.skip_track_ids ?? []),
    ...(overlay.overlay.quarantined_track_ids ?? []),
  ]);
  const trackIds = new Set(tracks.map((track) => track.trackId));

  for (const item of normalizedAppendTracks) {
    if (appendById.has(item.track_id)) {
      throw new Error(`Runtime overlay ${overlay.sourcePath} references duplicate append_tracks track_id: ${item.track_id}`);
    }
    if (trackIds.has(item.track_id)) {
      throw new Error(`Runtime overlay ${overlay.sourcePath} references append_tracks track_id already present in manifest: ${item.track_id}`);
    }
    appendById.set(item.track_id, item);
  }

  const knownTrackIds = new Set([...trackIds, ...appendById.keys()]);
  const skippedTrackIds: string[] = [];
  const resolvedTracks: RuntimeCampaignTrack[] = [];

  for (const track of tracks) {
    const runtimeOverride = normalizedById.get(track.trackId);
    const shouldSkip = skipTrackIds.includes(track.trackId) || runtimeOverride?.enabled === false;
    if (shouldSkip) {
      skippedTrackIds.push(track.trackId);
      continue;
    }

    resolvedTracks.push(applyRuntimeTrackOverride(track, runtimeOverride));
  }

  for (const appendTrack of normalizedAppendTracks) {
    const shouldSkip = skipTrackIds.includes(appendTrack.track_id) || appendTrack.enabled === false;
    if (shouldSkip) {
      skippedTrackIds.push(appendTrack.track_id);
      continue;
    }
    resolvedTracks.push(applyRuntimeTrackAppend(appendTrack));
  }

  const unknownSkippedIds = skipTrackIds.filter((trackId) => !knownTrackIds.has(trackId));
  if (unknownSkippedIds.length > 0) {
    throw new Error(
      `Runtime overlay ${overlay.sourcePath} references unknown skip_track_ids: ${unknownSkippedIds.join(", ")}`,
    );
  }

  const unknownOverrideIds = normalizedTracks
    .map((item) => item.track_id)
    .filter((trackId) => !trackIds.has(trackId));
  if (unknownOverrideIds.length > 0) {
    throw new Error(
      `Runtime overlay ${overlay.sourcePath} references unknown track_id overrides: ${unknownOverrideIds.join(", ")}`,
    );
  }

  return {
    tracks: resolvedTracks,
    skippedTrackIds,
  };
}

function parseRuntimeCampaignOverlay(text: string): RuntimeCampaignOverlay {
  const parsed = JSON.parse(text) as Record<string, unknown>;
  return {
    skip_track_ids: parseStringList(parsed.skip_track_ids),
    quarantined_track_ids: parseStringList(parsed.quarantined_track_ids),
    tracks: parseRuntimeTrackOverrides(parsed.tracks),
    append_tracks: parseRuntimeTrackOverrides(parsed.append_tracks),
  };
}

function parseRuntimeTrackOverrides(value: unknown): RuntimeTrackOverride[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item, index) => normalizeRuntimeTrackOverride(item, index))
    .filter((item): item is RuntimeTrackOverride => item !== null);
}

function normalizeRuntimeTrackOverride(value: unknown, index: number): RuntimeTrackOverride | null {
  if (!value || typeof value !== "object") {
    throw new Error(`Runtime overlay track override ${index} must be an object.`);
  }

  const record = value as Record<string, unknown>;
  const trackId = readRequiredString(record.track_id, `tracks[${index}].track_id`);
  return {
    track_id: trackId,
    enabled: typeof record.enabled === "boolean" ? record.enabled : undefined,
    track_goal: readOptionalString(record.track_goal),
    promotion_target: readOptionalString(record.promotion_target),
    smoke_command: readOptionalString(record.smoke_command),
    formal_command: readOptionalString(record.formal_command),
    allowed_change_scope: parseStringList(record.allowed_change_scope),
    internet_research_enabled: typeof record.internet_research_enabled === "boolean" ? record.internet_research_enabled : undefined,
    track_origin: readOptionalTrackOrigin(record.track_origin),
    force_fresh_thread: typeof record.force_fresh_thread === "boolean" ? record.force_fresh_thread : undefined,
  };
}

function applyRuntimeTrackOverride(track: CampaignTrack, override: RuntimeTrackOverride | undefined): RuntimeCampaignTrack {
  if (!override) {
    return cloneTrack(track);
  }

  return {
    ...track,
    trackGoal: override.track_goal ?? track.trackGoal,
    promotionTarget: override.promotion_target ?? track.promotionTarget,
    smokeCommand: override.smoke_command ?? track.smokeCommand,
    formalCommand: override.formal_command ?? track.formalCommand,
    allowedChangeScope: override.allowed_change_scope?.length ? [...override.allowed_change_scope] : [...track.allowedChangeScope],
    internetResearchEnabled: override.internet_research_enabled ?? track.internetResearchEnabled,
    trackOrigin: override.track_origin ?? (track as RuntimeCampaignTrack).trackOrigin,
    forceFreshThread: override.force_fresh_thread ?? (track as RuntimeCampaignTrack).forceFreshThread,
  };
}

function applyRuntimeTrackAppend(track: RuntimeTrackOverride): RuntimeCampaignTrack {
  if (!track.track_goal || !track.promotion_target || !track.smoke_command || !track.formal_command) {
    throw new Error(`Runtime overlay append_tracks entry ${track.track_id} must define track_goal, promotion_target, smoke_command, and formal_command.`);
  }

  return {
    trackId: track.track_id,
    trackGoal: track.track_goal,
    promotionTarget: track.promotion_target,
    smokeCommand: track.smoke_command,
    formalCommand: track.formal_command,
    allowedChangeScope: [...(track.allowed_change_scope ?? [])],
    internetResearchEnabled: track.internet_research_enabled ?? false,
    trackOrigin: track.track_origin ?? "incubation",
    forceFreshThread: track.force_fresh_thread ?? true,
  };
}

function cloneTrack(track: CampaignTrack): RuntimeCampaignTrack {
  return {
    ...track,
    allowedChangeScope: [...track.allowedChangeScope],
    trackOrigin: (track as RuntimeCampaignTrack).trackOrigin,
    forceFreshThread: (track as RuntimeCampaignTrack).forceFreshThread,
  };
}

function parseStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter((item): item is string => typeof item === "string" && item.trim() !== "")
    .map((item) => item.trim());
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

function readOptionalTrackOrigin(value: unknown): RuntimeTrackOrigin | undefined {
  if (value === "incubation") {
    return "incubation";
  }
  if (value === "default") {
    return "default";
  }
  return undefined;
}

function looksLikeInlineJson(value: string): boolean {
  return value.startsWith("{") || value.startsWith("[");
}

function resolveRepoPath(repoRoot: string, value: string): string {
  return path.isAbsolute(value) ? value : path.resolve(repoRoot, value);
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

function dedupeStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }
  return result;
}
