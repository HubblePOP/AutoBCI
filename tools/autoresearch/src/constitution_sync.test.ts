import assert from "node:assert/strict";
import test from "node:test";

import { collectChangedFilesFromInput, evaluateConstitutionSync } from "./constitution_sync.js";

test("evaluateConstitutionSync fails when a derived contract changes without CONSTITUTION.md", () => {
  const result = evaluateConstitutionSync([
    "tools/autoresearch/program.md",
    "README.md",
  ]);

  assert.equal(result.ok, false);
  assert.equal(result.requiresConstitutionUpdate, true);
  assert.match(result.message, /docs\/CONSTITUTION\.md/);
  assert.deepEqual(result.watchedFiles, ["tools/autoresearch/program.md"]);
});

test("evaluateConstitutionSync passes when CONSTITUTION.md changes together with tracked files", () => {
  const result = evaluateConstitutionSync([
    "docs/CONSTITUTION.md",
    "tools/autoresearch/src/run_campaign.ts",
  ]);

  assert.equal(result.ok, true);
  assert.equal(result.requiresConstitutionUpdate, true);
  assert.match(result.message, /in sync/i);
  assert.deepEqual(result.watchedFiles, ["tools/autoresearch/src/run_campaign.ts"]);
});

test("evaluateConstitutionSync ignores unrelated file changes", () => {
  const result = evaluateConstitutionSync([
    "reports/2026-04-07/experiment_status.md",
    "scripts/train_tree_baseline.py",
  ]);

  assert.equal(result.ok, true);
  assert.equal(result.requiresConstitutionUpdate, false);
  assert.equal(result.watchedFiles.length, 0);
});

test("evaluateConstitutionSync normalizes absolute paths against the repo root", () => {
  const repoRoot = "/repo";
  const result = evaluateConstitutionSync([
    "/repo/docs/CONSTITUTION.md",
    "/repo/tools/autoresearch/src/run_campaign.ts",
  ], repoRoot);

  assert.equal(result.ok, true);
  assert.deepEqual(result.watchedFiles, ["tools/autoresearch/src/run_campaign.ts"]);
});

test("collectChangedFilesFromInput prefers explicit CLI paths", () => {
  const files = collectChangedFilesFromInput({
    argv: ["tools/autoresearch/program.md", "docs/CONSTITUTION.md"],
    stdinText: "tools/autoresearch/src/run_campaign.ts\n",
  });

  assert.deepEqual(files, [
    "tools/autoresearch/program.md",
    "docs/CONSTITUTION.md",
  ]);
});

test("collectChangedFilesFromInput falls back to newline-delimited stdin", () => {
  const files = collectChangedFilesFromInput({
    argv: [],
    stdinText: "tools/autoresearch/program.current.md\nREADME.md\n\n",
  });

  assert.deepEqual(files, [
    "tools/autoresearch/program.current.md",
    "README.md",
  ]);
});
