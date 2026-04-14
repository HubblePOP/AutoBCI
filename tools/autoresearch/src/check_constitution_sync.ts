import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { collectChangedFilesFromInput, evaluateConstitutionSync } from "./constitution_sync.js";

async function main() {
  const stdinText = process.stdin.isTTY ? "" : await readStdin();
  const changedFiles = collectChangedFilesFromInput({
    argv: process.argv.slice(2),
    stdinText,
  });
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(scriptDir, "..", "..", "..");
  const result = evaluateConstitutionSync(changedFiles, repoRoot);

  const output = [
    result.message,
    result.watchedFiles.length > 0 ? `watched_files=${result.watchedFiles.join(",")}` : "watched_files=",
  ].join("\n");

  if (result.ok) {
    process.stdout.write(`${output}\n`);
    return;
  }

  process.stderr.write(`${output}\n`);
  process.exitCode = 1;
}

async function readStdin(): Promise<string> {
  const chunks: string[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(typeof chunk === "string" ? chunk : chunk.toString("utf8"));
  }
  return chunks.join("");
}

await main();
