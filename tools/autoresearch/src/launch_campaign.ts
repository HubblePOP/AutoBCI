import { main as runCampaignMain } from "./run_campaign.js";
import { stripLaunchEnvironment } from "./launch_support.js";
import { compileAutoresearchQueue, shouldCompileAutoresearchQueue } from "./queue_compiler.js";
import path from "node:path";
import { pathToFileURL } from "node:url";

type LaunchCampaignOptions = {
  stripEnvironment?: () => void;
  runCampaign?: () => Promise<void>;
  compileQueue?: () => Promise<void>;
};

export async function launchCampaign(options: LaunchCampaignOptions = {}): Promise<void> {
  const stripEnvironment = options.stripEnvironment ?? (() => stripLaunchEnvironment());
  const runCampaign = options.runCampaign ?? runCampaignMain;
  const compileQueue = options.compileQueue ?? (async () => {
    if (shouldCompileAutoresearchQueue(process.argv.slice(2))) {
      try {
        await compileAutoresearchQueue();
      } catch (error) {
        console.warn(`queue compiler failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  });

  stripEnvironment();
  await compileQueue();
  await runCampaign();
}

function isEntrypoint(): boolean {
  const argv1 = process.argv[1];
  if (!argv1) {
    return false;
  }
  return pathToFileURL(path.resolve(argv1)).href === import.meta.url;
}

if (isEntrypoint()) {
  await launchCampaign();
}
