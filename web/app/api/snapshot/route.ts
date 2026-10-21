import seedSnapshot from "../../seed-snapshot.json";
import {
  expectedSyncToken,
  readLatestSnapshot,
  saveLatestSnapshot,
} from "../../../db/panel-snapshot";

export const dynamic = "force-dynamic";

const privateKeys = new Set([
  "command",
  "commands",
  "dataset",
  "job_id",
  "log",
  "log_tail",
  "logs",
  "manifest",
  "run_root",
  "scheduler_id",
  "source",
]);
const privateText = ["/home/", "/project/", "/scratch/", "sbatch ", "srun ", "ssh "];

function noStore(payload: unknown, init: ResponseInit = {}) {
  const headers = new Headers(init.headers);
  headers.set("Cache-Control", "no-store");
  return Response.json(payload, { ...init, headers });
}

function containsPrivateCampaignData(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsPrivateCampaignData);
  if (value && typeof value === "object") {
    return Object.entries(value).some(([key, child]) =>
      privateKeys.has(key.toLowerCase()) || containsPrivateCampaignData(child)
    );
  }
  if (typeof value === "string") {
    const lowered = value.toLowerCase();
    return privateText.some((fragment) => lowered.includes(fragment));
  }
  return false;
}

export async function GET(request: Request) {
  if (!isLocalRequest(request) && !hasChatGPTIdentity(request)) {
    return noStore({ error: "Sign in to view campaign results" }, { status: 401 });
  }
  try {
    const snapshot = await readLatestSnapshot();
    return noStore(snapshot ?? seedSnapshot);
  } catch {
    return noStore(seedSnapshot);
  }
}

export async function POST(request: Request) {
  const expected = expectedSyncToken();
  const supplied = request.headers.get("authorization") ?? "";
  if (!expected || supplied !== "Bearer " + expected) {
    return noStore({ error: "Not authorized" }, { status: 401 });
  }

  const length = Number(request.headers.get("content-length") ?? "0");
  if (length > 8_000_000) {
    return noStore({ error: "Snapshot is too large" }, { status: 413 });
  }

  let snapshot: unknown;
  try {
    snapshot = await request.json();
  } catch {
    return noStore({ error: "Snapshot must be JSON" }, { status: 400 });
  }
  if (
    !snapshot ||
    typeof snapshot !== "object" ||
    (snapshot as { schema?: string }).schema !== "pde_panel_snapshot_v1" ||
    !Array.isArray((snapshot as { runs?: unknown }).runs)
  ) {
    return noStore({ error: "Unsupported snapshot" }, { status: 400 });
  }
  if (containsPrivateCampaignData(snapshot)) {
    return noStore({ error: "Snapshot contains server-only data" }, { status: 400 });
  }

  await saveLatestSnapshot(snapshot);
  return noStore({ ok: true });
}

function isLocalRequest(request: Request) {
  const hostname = new URL(request.url).hostname;
  return hostname === "localhost" || hostname === "127.0.0.1";
}

function hasChatGPTIdentity(request: Request) {
  return Boolean(request.headers.get("oai-authenticated-user-id"));
}
