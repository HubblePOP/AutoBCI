import path from "node:path";

const CONSTITUTION_PATH = "docs/CONSTITUTION.md";
const WATCHED_FILES = new Set([
  "tools/autoresearch/program.md",
  "tools/autoresearch/program.current.md",
  "tools/autoresearch/tracks.current.json",
  "tools/autoresearch/src/run_campaign.ts",
]);

export interface ConstitutionSyncResult {
  ok: boolean;
  requiresConstitutionUpdate: boolean;
  watchedFiles: string[];
  message: string;
}

export function collectChangedFilesFromInput({
  argv,
  stdinText,
}: {
  argv: string[];
  stdinText: string;
}): string[] {
  const source = argv.length > 0 ? argv : stdinText.split(/\r?\n/);
  return source
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "");
}

export function evaluateConstitutionSync(changedFiles: string[], repoRoot?: string): ConstitutionSyncResult {
  const normalized = changedFiles
    .map((file) => normalizePath(file, repoRoot))
    .filter((file): file is string => file !== null);
  const uniqueFiles = Array.from(new Set(normalized));
  const watchedFiles = uniqueFiles.filter((file) => WATCHED_FILES.has(file));
  const constitutionChanged = uniqueFiles.includes(CONSTITUTION_PATH);

  if (watchedFiles.length === 0) {
    return {
      ok: true,
      requiresConstitutionUpdate: false,
      watchedFiles: [],
      message: "No tracked AutoResearch contract files changed.",
    };
  }

  if (!constitutionChanged) {
    return {
      ok: false,
      requiresConstitutionUpdate: true,
      watchedFiles,
      message: `Conceptual AutoResearch files changed (${watchedFiles.join(", ")}), but ${CONSTITUTION_PATH} was not updated.`,
    };
  }

  return {
    ok: true,
    requiresConstitutionUpdate: true,
    watchedFiles,
    message: `AutoResearch contract files and ${CONSTITUTION_PATH} are in sync.`,
  };
}

function normalizePath(filePath: string, repoRoot?: string): string | null {
  if (filePath.trim() === "") {
    return null;
  }
  const normalized = filePath.replaceAll("\\", "/");
  if (!repoRoot || !path.isAbsolute(filePath)) {
    return trimLeadingDot(normalized);
  }
  const relative = path.relative(repoRoot, filePath).replaceAll("\\", "/");
  return trimLeadingDot(relative);
}

function trimLeadingDot(filePath: string): string {
  return filePath.replace(/^\.\//, "");
}
