import assert from "node:assert/strict";
import test from "node:test";

import { sanitizeLaunchEnvironment } from "./launch_support.js";

test("sanitizeLaunchEnvironment removes npm_config_prefix without mutating the original env", () => {
  const original = {
    HOME: "/tmp/home",
    npm_config_prefix: "/tmp/npm-prefix",
  } as NodeJS.ProcessEnv;

  const sanitized = sanitizeLaunchEnvironment(original);

  assert.equal(sanitized.HOME, "/tmp/home");
  assert.equal(sanitized.npm_config_prefix, undefined);
  assert.equal(original.npm_config_prefix, "/tmp/npm-prefix");
});
