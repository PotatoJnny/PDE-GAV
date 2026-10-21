import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";

async function worker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", String(process.pid) + "-" + String(Date.now()));
  return (await import(workerUrl.href)).default;
}

const bindings = {
  ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
};

const context = {
  waitUntil() {},
  passThroughOnException() {},
};

test("renders the PDE Agent campaign product", async () => {
  const app = await worker();
  const response = await app.fetch(
    new Request("http://localhost/", {
      headers: {
        accept: "text/html",
        "oai-authenticated-user-id": "test-user",
        "oai-authenticated-user-email": "scientist@example.com",
      },
    }),
    bindings,
    context,
  );
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>PDE Agent<\/title>/i);
  assert.match(html, /PDE Agent/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("seed snapshot contains product views without cluster paths", async () => {
  const source = await readFile(new URL("../app/seed-snapshot.json", import.meta.url), "utf8");
  const snapshot = JSON.parse(source);
  const dashboard = await readFile(new URL("../app/dashboard.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(dashboard, /seed-snapshot\.json/);
  assert.equal(snapshot.schema, "pde_panel_snapshot_v1");
  assert.ok(Array.isArray(snapshot.runs));
  assert.ok(snapshot.runs.length >= 1);
  const serialized = JSON.stringify(snapshot);
  assert.doesNotMatch(serialized, /"run_root"|"job_id"|"log_tail"|"commands"|"source"/);
  assert.doesNotMatch(serialized, /\/home\/|\/scratch\/|\/project\//);
});

test("dashboard reads require GPT identity and writes require a token", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const source = await readFile(new URL("../app/api/snapshot/route.ts", import.meta.url), "utf8");
  assert.match(page, /requireChatGPTUser/);
  assert.match(source, /oai-authenticated-user-id/);
  assert.match(source, /authorization/);
  assert.match(source, /containsPrivateCampaignData/);
});

test("social preview is project-specific", async () => {
  const details = await stat(new URL("../public/og.png", import.meta.url));
  assert.ok(details.size > 100_000);
});
