import assert from "node:assert/strict";
import test from "node:test";

import { launchCampaign } from "./launch_campaign.js";

test("launchCampaign strips launch env before invoking the campaign entrypoint", async () => {
  const events: string[] = [];

  await launchCampaign({
    stripEnvironment: () => {
      events.push("strip");
    },
    compileQueue: async () => {
      events.push("compile");
    },
    runCampaign: async () => {
      events.push("run");
    },
  });

  assert.deepEqual(events, ["strip", "compile", "run"]);
});
