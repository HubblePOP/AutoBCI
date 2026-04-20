import { readFileSync, writeFileSync } from "node:fs";
import { Codex } from "@openai/codex-sdk";

const promptPath = process.env.DIRECTOR_PROMPT_PATH || "";
const outputPath = process.env.DIRECTOR_OUTPUT_PATH || "";
const repoRoot = process.env.DIRECTOR_REPO_ROOT || process.cwd();
const threadId = process.env.DIRECTOR_THREAD_ID || "";

if (!promptPath || !outputPath) {
  console.error("DIRECTOR_PROMPT_PATH and DIRECTOR_OUTPUT_PATH are required");
  process.exit(1);
}

const prompt = readFileSync(promptPath, "utf-8");
const codex = new Codex({ config: { model_reasoning_effort: "high" } });

const threadOpts = {
  workingDirectory: repoRoot,
  skipGitRepoCheck: true,
  sandboxMode: "workspace-write",
  approvalPolicy: "never",
  networkAccessEnabled: true,
};

async function main() {
  const thread = threadId
    ? codex.resumeThread(threadId, threadOpts)
    : codex.startThread(threadOpts);
  const turn = await thread.run(prompt);
  writeFileSync(outputPath, turn.finalResponse, "utf-8");
  console.log("DIRECTOR_THREAD_ID=" + (thread.id || ""));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
