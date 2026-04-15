import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

export type ExpectedMemoryClass = "high" | "low" | "service";

export interface ManagedProcessEntry {
  pid: number;
  campaignId: string;
  trackId: string | null;
  taskKind: string;
  modelFamily: string | null;
  priority: number;
  expectedMemoryClass: ExpectedMemoryClass;
  command: string;
  startedAt: string;
  cwd: string;
}

export interface ProcessRegistryEntry extends ManagedProcessEntry {
  alive?: boolean;
  rssMb?: number | null;
}

export interface ProcessRegistry {
  updatedAt: string;
  processes: ProcessRegistryEntry[];
}

export function classifyExpectedMemoryClass(runnerFamily: string | undefined): ExpectedMemoryClass {
  const normalized = (runnerFamily || "").toLowerCase();
  if (
    normalized === "feature_lstm"
    || normalized === "feature_gru"
    || normalized === "feature_tcn"
    || normalized === "raw_lstm"
  ) {
    return "high";
  }
  if (normalized === "dashboard" || normalized === "supervisor" || normalized === "service") {
    return "service";
  }
  return "low";
}

export async function readProcessRegistry(registryPath: string): Promise<ProcessRegistry> {
  try {
    const text = await readFile(registryPath, "utf8");
    const parsed = JSON.parse(text) as Partial<ProcessRegistry>;
    return {
      updatedAt: typeof parsed.updatedAt === "string" ? parsed.updatedAt : "",
      processes: Array.isArray(parsed.processes) ? parsed.processes : [],
    };
  } catch {
    return {
      updatedAt: "",
      processes: [],
    };
  }
}

export async function registerManagedProcess(
  registryPath: string,
  entry: ManagedProcessEntry,
): Promise<void> {
  const registry = await readProcessRegistry(registryPath);
  const withoutPid = registry.processes.filter((item) => item.pid !== entry.pid);
  withoutPid.push({ ...entry });
  await writeProcessRegistry(registryPath, {
    updatedAt: new Date().toISOString(),
    processes: withoutPid.sort((a, b) => a.pid - b.pid),
  });
}

export async function unregisterManagedProcess(registryPath: string, pid: number): Promise<void> {
  const registry = await readProcessRegistry(registryPath);
  await writeProcessRegistry(registryPath, {
    updatedAt: new Date().toISOString(),
    processes: registry.processes.filter((item) => item.pid !== pid),
  });
}

async function writeProcessRegistry(registryPath: string, registry: ProcessRegistry): Promise<void> {
  await mkdir(path.dirname(registryPath), { recursive: true });
  const tmpPath = `${registryPath}.tmp`;
  await writeFile(tmpPath, `${JSON.stringify(registry, null, 2)}\n`, "utf8");
  await rename(tmpPath, registryPath);
}
