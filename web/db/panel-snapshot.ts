import { env } from "cloudflare:workers";

interface D1Statement {
  bind(...values: unknown[]): D1Statement;
  first<T>(): Promise<T | null>;
  run(): Promise<unknown>;
}

interface D1DatabaseLike {
  prepare(sql: string): D1Statement;
}

interface PanelBindings {
  DB?: D1DatabaseLike;
  PDE_PANEL_SYNC_TOKEN?: string;
}

const CREATE_TABLE = "CREATE TABLE IF NOT EXISTS panel_snapshots (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL)";

function bindings(): PanelBindings {
  return env as unknown as PanelBindings;
}

async function database(): Promise<D1DatabaseLike> {
  const db = bindings().DB;
  if (!db) {
    throw new Error("Panel database is unavailable");
  }
  await db.prepare(CREATE_TABLE).run();
  return db;
}

export function expectedSyncToken(): string {
  return bindings().PDE_PANEL_SYNC_TOKEN ?? "";
}

export async function readLatestSnapshot(): Promise<unknown | null> {
  const db = await database();
  const row = await db
    .prepare("SELECT payload FROM panel_snapshots WHERE id = ?")
    .bind("latest")
    .first<{ payload: string }>();
  if (!row) {
    return null;
  }
  return JSON.parse(row.payload);
}

export async function saveLatestSnapshot(snapshot: unknown): Promise<void> {
  const db = await database();
  await db
    .prepare(
      "INSERT INTO panel_snapshots (id, payload, updated_at) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at"
    )
    .bind("latest", JSON.stringify(snapshot), new Date().toISOString())
    .run();
}
