export function sanitizeLaunchEnvironment(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const cleaned: NodeJS.ProcessEnv = { ...env };
  delete cleaned.npm_config_prefix;
  return cleaned;
}

export function stripLaunchEnvironment(env: NodeJS.ProcessEnv = process.env): void {
  delete env.npm_config_prefix;
}
