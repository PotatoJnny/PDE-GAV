"""Dependency-free browser panel for the first Toy PDE slice."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping
from urllib.parse import urlparse

from .controller import PanelController, build_toy_panel


def run_panel(host: str = "127.0.0.1", port: int = 8000, *, controller: PanelController | None = None) -> None:
    controller = controller if controller is not None else build_toy_panel()
    handler = _handler(controller)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"PDE Agent Verifier Solver panel: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler(controller: PanelController):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, HTML, "text/html; charset=utf-8")
                return
            if path == "/api/setup":
                self._json(200, controller.setup_page())
                return
            if path.startswith("/api/runs/"):
                run_id = path.rsplit("/", 1)[-1]
                try:
                    page = controller.run_page(run_id)
                except KeyError as exc:
                    self._json(404, {"error": str(exc)})
                    return
                self._json(200, page)
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in ("/api/runs", "/api/prompt-preview"):
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                values = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(values, Mapping):
                    raise ValueError("request body must be an object")
                if path == "/api/prompt-preview":
                    self._json(200, controller.prompt_preview(values))
                    return
                response = controller.create_and_launch(values)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(202, response)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, status: int, value: object) -> None:
            self._send(status, json.dumps(value).encode(), "application/json")

        def _send(self, status: int, body: str | bytes, content_type: str) -> None:
            encoded = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PDE Agent Verifier Solver</title>
  <style>
    :root { --ink:#17231d; --muted:#647168; --paper:#f4f1e9; --card:#fffdf7;
      --line:#d9d5c8; --green:#19734a; --blue:#315e81; --orange:#b65c2b; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--paper);
      font:15px/1.45 ui-sans-serif,system-ui,sans-serif; }
    main { max-width:1120px; margin:0 auto; padding:36px 24px 80px; }
    h1 { margin:0 0 8px; font:700 38px/1.1 ui-serif,Georgia,serif; }
    h2 { margin:0 0 18px; font-size:22px; } h3 { margin:0 0 8px; }
    p { color:var(--muted); } .lead { max-width:760px; font-size:17px; }
    .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
      padding:18px; box-shadow:0 3px 16px #1f2c2210; }
    section { margin-top:28px; } form .grid { grid-template-columns:repeat(4,1fr); }
    label { display:block; color:var(--muted); font-size:12px; font-weight:700;
      letter-spacing:.04em; text-transform:uppercase; }
    input,select,textarea { width:100%; margin-top:6px; border:1px solid var(--line);
      border-radius:8px; padding:10px; background:white; color:var(--ink); font:inherit; }
    textarea { min-height:74px; resize:vertical; }
    .check { display:flex; align-items:center; gap:8px; margin-top:24px; color:var(--ink);
      text-transform:none; letter-spacing:0; font-size:14px; }
    .check input { width:auto; margin:0; }
    button { margin-top:18px; border:0; border-radius:9px; padding:11px 18px;
      background:var(--green); color:white; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    .status { display:inline-block; border-radius:999px; padding:5px 10px;
      background:#e3ece6; color:var(--green); font-weight:700; }
    .events { max-height:300px; overflow:auto; }
    .event { padding:9px 0; border-bottom:1px solid var(--line); display:flex; gap:12px; }
    .event small { min-width:70px; color:var(--muted); }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:9px; text-align:left; border-bottom:1px solid var(--line); }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; }
    .memory { display:flex; flex-direction:column; gap:14px; }
    .memory-round { border:1px solid var(--line); border-radius:12px; padding:14px;
      background:#fbfaf4; }
    .memory-agents,.candidate-history { display:grid; grid-template-columns:repeat(2,1fr);
      gap:10px; }
    .memory-entry,.controller-entry { border:1px solid var(--line); border-radius:10px;
      padding:12px; background:var(--card); min-width:0; }
    .controller-entry { margin-top:10px; border-color:#b7c8bb; background:#f4faf5; }
    .memory-meta { color:var(--muted); font-size:12px; }
    .memory-log { margin:10px 0 0; white-space:pre-wrap; overflow-wrap:anywhere;
      font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .candidate-history .selected { border-color:var(--green);
      box-shadow:inset 4px 0 var(--green); }
    .promotion-evidence { margin-top:18px; padding-top:16px; border-top:1px solid var(--line); }
    .promotion-reports { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }
    .promotion-report { background:var(--card); border:1px solid var(--line);
      border-radius:10px; padding:12px; overflow:auto; }
    .promotion-report h3 { display:flex; justify-content:space-between; gap:10px; }
    .result { border-color:var(--green); background:#f7fff8; }
    code { background:#edf0ea; padding:2px 5px; border-radius:4px; }
    .error { color:#a52b25; white-space:pre-wrap; } [hidden] { display:none !important; }
    @media(max-width:800px) { .grid,form .grid,.memory-agents,.candidate-history {
      grid-template-columns:1fr; } }
  </style>
</head>
<body><main>
  <header><h1>PDE Agent Verifier Solver</h1>
    <p class="lead">Configure one Generator–Agent–Verifier campaign, watch every round,
    and receive the selected scientific program. This first slice runs a real 1D Poisson example.</p>
  </header>

  <section id="programs" class="grid"></section>

  <section class="card"><h2>Setup</h2>
    <form id="setup-form">
      <div class="grid">
        <label>Project name<input id="project" value="Toy Poisson campaign"></label>
        <label>Dataset type<select id="dataset"><option value="numerical">Numerical coarse fields</option>
          <option value="experimental">Noisy observations</option></select></label>
        <label>Dataset path (optional)<input id="dataset-location" placeholder="/path/to/data"></label>
        <label>Task<select><option>Full-state solver</option></select></label>
        <label>Development sampling<select id="sampling"><option value="agent_adaptive">Agent chooses Cases</option>
          <option value="fixed_recipe">Fixed recipe (ablation)</option></select></label>
        <label>Rounds<input id="rounds" type="number" min="1" value="3"></label>
        <label>Master seed<input id="master-seed" type="number" min="0" value="0"></label>
        <label>Development coordinates<select id="development-coordinate-mode">
          <option value="private_random_points">Private random points</option>
          <option value="user_fixed_query">User-fixed grid</option>
          <option value="agent_selected_query">Agent-selected query (explicit opt-in)</option>
        </select></label>
        <label>Private random point count<input id="development-point-count" type="number" min="1" value="32"></label>
        <label>User-fixed grid x resolution<input id="grid-x" type="number" min="1" value="65"></label>
        <label>Max sampled / Agent / round (blank = unlimited)<input id="sample-limit" type="number" min="1" value="12" placeholder="Unlimited"></label>
        <label>Agent total time per round (seconds, including tools)<input id="budget" type="number" min="1" value="300"></label>
        <label>Round hidden: original<input id="round-original" type="number" min="0" value="1"></label>
        <label>Round hidden: base<input id="round-base" type="number" min="0" value="3"></label>
        <label>Round hidden: guided<input id="round-guided" type="number" min="0" value="2"></label>
        <label>Terminal base cohort: original<input id="terminal-original" type="number" min="0" value="1"></label>
        <label>Terminal base cohort: base<input id="terminal-base" type="number" min="0" value="4"></label>
        <label>Terminal rank by<select id="terminal-distribution"><option value="base">Base</option><option value="original">Original</option></select></label>
        <label>Terminal statistic<select id="terminal-statistic"><option value="mean">Mean error</option><option value="cvar">Tail / CVaR</option><option value="maximum">Maximum error</option></select></label>
        <label>Promote by<select id="promote-distribution"><option value="base">Base</option><option value="original">Original</option></select></label>
        <label>Promotion statistic<select id="promote-statistic"><option value="mean">Mean error</option><option value="cvar">Tail / CVaR</option><option value="maximum">Maximum error</option></select></label>
        <label class="check"><input id="allow-grid-truth" type="checkbox"> Allow grid truth on explicit request</label>
        <label class="check"><input id="allow-point-truth" type="checkbox"> Allow point truth on explicit request</label>
      </div>
      <h3 style="margin-top:22px">Plugin settings</h3>
      <div id="plugin-settings" class="grid"></div>
      <label>Round-end submission selection<select id="nomination-policy">
        <option value="">Use each Agent's configured rule</option>
        <option value="single_primary_v1">One explicitly selected primary per Agent</option>
        <option value="latest">Last submitted version (legacy)</option>
        <option value="all">All submitted versions</option>
      </select></label>
      <p id="submission-rule"></p>
      <p id="evaluation-resources"></p>
      <h3 style="margin-top:22px">Agent instructions</h3>
      <p>Write or load Markdown for the task and available resources. Each Agent can have its own broad research direction. Run budgets and tool permissions come from the configuration.</p>
      <label>Task<textarea id="prompt-task">Develop the strongest scientific program for this task using the available environment.</textarea><input type="file" accept=".md,.txt,text/markdown" data-prompt-file="prompt-task"></label>
      <label>Available resources<textarea id="prompt-resources" placeholder="Describe the public data, starting program and useful references."></textarea><input type="file" accept=".md,.txt,text/markdown" data-prompt-file="prompt-resources"></label>
      <div id="prompt-roles" class="grid"></div>
      <button type="button" id="preview-prompt">Preview instructions</button>
      <p id="prompt-preview-note"></p><div id="prompt-preview"></div>
      <p>The default lets each Agent choose fresh Cases up to one limit. Fixed recipe uses
        the same sampler and exists only for controlled ablations.</p>
      <button id="launch">Launch run</button> <span id="form-error" class="error"></span>
    </form>
  </section>

  <section id="live" hidden><div class="card"><h2>Live Run <span id="status" class="status"></span></h2>
    <div id="events" class="events"></div></div>
    <div class="card" style="margin-top:14px"><h2>Round results</h2><div id="rounds-view"></div></div>
    <div class="card" style="margin-top:14px"><h2>Campaign memory</h2>
      <p>Each round keeps both Agents' own logs followed by the Controller's official result.</p>
      <div id="memory" class="memory"></div></div>
    <div class="card" style="margin-top:14px"><h2>Candidate history</h2>
      <p>The ancestry and selection history of submitted scientific programs.</p>
      <div id="candidate-history" class="candidate-history"></div></div>
    <div id="result" class="card result" style="margin-top:14px" hidden></div>
  </section>

<script>
const $ = id => document.getElementById(id);
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  })[character]);
}
function campaignMemoryMarkup(memory) {
  const rounds=memory?.rounds||[];
  if (!rounds.length) return 'Campaign memory will appear after the first finished or aborted round.';
  return rounds.map(round=>{
    const agents=(round.agents||[]).map(log=>`<section class="memory-entry">
      <h3>${escapeHtml(log.agent_name)}</h3>
      <div class="memory-meta">${escapeHtml(log.agent_id)} · ${escapeHtml(log.status)}</div>
      <pre class="memory-log">${escapeHtml(log.markdown)}</pre></section>`).join('');
    const roundStatus=round.status||"completed";
    return `<article class="memory-round"><h3>Round ${Number(round.round)} · ${escapeHtml(roundStatus)}</h3>
      <div class="memory-agents">${agents}</div>
      <section class="controller-entry"><h3>Controller outcome</h3>
      <pre class="memory-log">${escapeHtml(round.controller_markdown)}</pre></section></article>`;
  }).join('');
}
function candidateHistoryMarkup(nodes) {
  if (!nodes.length) return 'Candidate history will appear after the first submission.';
  return nodes.map(node=>{
    const parent=node.parent ? `Parent ${escapeHtml(node.parent)}` : 'Root candidate';
    const distributions=Object.entries(node.official_distributions||{}).map(([name,row])=>{
      const mean=Number(row&&row.mean);
      const cvar=Number(row&&row.cvar);
      const maximum=Number(row&&row.maximum);
      return `<div><strong>${escapeHtml(name)}</strong>: mean ${Number.isFinite(mean)?mean.toFixed(4):'—'}, `+
        `CVaR ${Number.isFinite(cvar)?cvar.toFixed(4):'—'}, max ${Number.isFinite(maximum)?maximum.toFixed(4):'—'}</div>`;
    }).join('') || '<div>Official evaluation not completed.</div>';
    const failures=(node.failures||[]).map(value=>`<div>${escapeHtml(value)}</div>`).join('');
    return `<article class="card ${node.selected?'selected':''}">
      <div class="memory-meta">Round ${Number(node.round)} · ${escapeHtml(node.agent_id)}
        · ${escapeHtml(node.status)}</div>
      <h3>${escapeHtml(node.title)}</h3><p>${escapeHtml(node.summary)}</p>
      <div>${distributions}${failures}</div><div class="memory-meta">${parent}</div></article>`;
  }).join('');
}
function fieldMarkup(group, field) {
  const attrs = `data-group="${escapeHtml(group)}" data-name="${escapeHtml(field.name)}" data-type="${escapeHtml(field.type)}"`;
  if (field.type === 'boolean') {
    return `<label class="check"><input ${attrs} type="checkbox" ${field.default?'checked':''}> ${escapeHtml(field.label)}</label>`;
  }
  if (field.choices && field.choices.length) {
    const options = field.choices.map(([value,label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join('');
    return `<label>${escapeHtml(field.label)}<select ${attrs}>${options}</select></label>`;
  }
  const type = field.type === 'integer' ? 'number' : 'text';
  return `<label>${escapeHtml(field.label)}<input ${attrs} type="${type}" value="${escapeHtml(field.default??'')}"></label>`;
}
function collectSettings(group) {
  const values = {};
  document.querySelectorAll(`[data-group="${group}"]`).forEach(input => {
    let value = input.value;
    if (input.dataset.type === 'boolean') value = input.checked;
    else if (input.dataset.type === 'integer') value = Number(input.value);
    values[input.dataset.name] = value;
  });
  return values;
}
function promptValues() {
  return {agent_prompt_sections:{task:$('prompt-task').value, resources:$('prompt-resources').value},
    agent_roles:Object.fromEntries([...document.querySelectorAll('[data-prompt-role]')].map(el => [el.dataset.promptRole, el.value]))};
}
$('setup-form').addEventListener('change', async event => {
  const target = event.target.dataset.promptFile;
  if (target && event.target.files?.length) $(target).value = await event.target.files[0].text();
});
$('preview-prompt').addEventListener('click', async () => {
  const response = await fetch('/api/prompt-preview',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({...promptValues(),agents:setupData?.defaults?.agents || [],task_settings:collectSettings('task')})});
  const body = await response.json();
  $('prompt-preview-note').textContent = body.error || body.note;
  $('prompt-preview').innerHTML = response.ok ? body.agents.map(a =>
    `<details><summary>${escapeHtml(a.name)}</summary><pre style="white-space:pre-wrap">${escapeHtml(a.prompt)}</pre></details>`).join('') : '';
});
let setupData = null;
async function setup() {
  const data = await fetch('/api/setup').then(r => r.json());
  setupData = data;
  $('nomination-policy').disabled = !data.submission_selection_available;
  $('submission-rule').textContent = submissionRuleText(data.agent_submission_policies);
  $('evaluation-resources').textContent = evaluationResourceText(data.candidate_execution);
  $('programs').innerHTML = [data.problem,data.task,data.generator].map(p =>
    `<article class="card"><h3>${escapeHtml(p.title)}</h3><p>${escapeHtml(p.description)}</p></article>`).join('');
  const titles = {problem:'Problem',task:'Task',generator:'Generator'};
  $('plugin-settings').innerHTML = Object.entries(data.forms).map(([group,fields]) =>
    `<div class="card"><h3>${titles[group]}</h3>${fields.map(field =>
      fieldMarkup(group,field)).join('')}</div>`).join('');
  const defaults = data.defaults || {};
  $('prompt-roles').innerHTML = (defaults.agents || []).map((id,index) =>
    `<label>${escapeHtml(id)} direction<textarea id="prompt-role-${index}" data-prompt-role="${escapeHtml(id)}" placeholder="Optional broad research direction"></textarea><input type="file" accept=".md,.txt,text/markdown" data-prompt-file="prompt-role-${index}"></label>`).join('');
  const ids = {project_name:'project',round_count:'rounds',master_seed:'master-seed',
    agent_budget_seconds:'budget',max_sampled_cases_per_agent_per_round:'sample-limit',
    development_point_count:'development-point-count',development_coordinate_mode:'development-coordinate-mode'};
  Object.entries(ids).forEach(([key,id]) => { if (defaults[key] !== undefined) $(id).value=defaults[key] ?? ''; });
  const counts = defaults.round_hidden || {};
  ['original','base','guided'].forEach(name => { $('round-'+name).value=counts[name] || 0; });
  const terminal = defaults.terminal_evaluation?.cohorts?.[0]?.counts || {};
  ['original','base'].forEach(name => { $('terminal-'+name).value=terminal[name] || 0; });
  if (data.backend?.prepared) {
    document.querySelectorAll('[data-group="problem"]').forEach(input => { input.disabled=true; });
    $('dataset-location').disabled=true;
    $('dataset').disabled=true;
    $('grid-x').disabled=true;
    if (!(defaults.allowed_sources || []).includes('original')) {
      $('round-original').disabled=true; $('terminal-original').disabled=true;
    }
    if (!data.backend.guidance) $('round-guided').disabled=true;
    document.querySelector('.lead').textContent =
      'Run the loaded scientific backend in the unified campaign. Candidate outputs: '+
      (data.backend.query_fields || []).join(', ')+'. Physical assets are fixed for this session.';
  }
}
$('setup-form').addEventListener('submit', async event => {
  event.preventDefault(); $('launch').disabled = true; $('form-error').textContent = '';
  const bound = !!setupData?.backend?.prepared;
  const defaults = setupData?.defaults || {};
  const problemSettings = bound ? {} : collectSettings('problem');
  const taskSettings = collectSettings('task');
  const generatorSettings = collectSettings('generator');
  const guided = generatorSettings.guidance_enabled ?? defaults.guidance_enabled ?? true;
  const payload = {
    ...defaults,
    ...(bound ? {allowed_sources:(defaults.allowed_sources || ['base']).filter(source => guided || source !== 'guided')} : {}),
    project_name:$('project').value,
    dataset_kind:$('dataset').value,
    dataset_location:$('dataset-location').value,
    sampling_strategy:$('sampling').value, round_count:+$('rounds').value,
    master_seed:+$('master-seed').value,
    development_coordinate_mode:$('development-coordinate-mode').value,
    development_point_count:+$('development-point-count').value,
    grid_x:+$('grid-x').value,
    allow_grid_truth:$('allow-grid-truth').checked,
    allow_point_truth:$('allow-point-truth').checked,
    max_sampled_cases_per_agent_per_round:$('sample-limit').value.trim() === '' ? null : +$('sample-limit').value,
    agent_budget_seconds:+$('budget').value,
    ...($('nomination-policy').value ? {nomination_policy:$('nomination-policy').value} : {}),
    problem_settings:problemSettings,
    task_settings:taskSettings,
    generator_settings:generatorSettings,
    ...promptValues(), agents:defaults.agents || [],
    starter_cases:defaults.starter_cases || {original:2,base:2,guided:0},
    fixed_recipe:bound ? defaults.fixed_recipe : {original:4,base:guided?6:8,guided:guided?2:0},
    round_hidden:{original:+$('round-original').value,base:+$('round-base').value,
      guided:guided?+$('round-guided').value:0},
    terminal_evaluation:{
      cohorts:[{id:'base_model',guidance_source:'base',counts:{
        original:+$('terminal-original').value,base:+$('terminal-base').value}}],
      ranking:{cohort_id:'base_model',distribution:$('terminal-distribution').value,
        statistic:$('terminal-statistic').value}},
    promotion:{distribution:$('promote-distribution').value,statistic:$('promote-statistic').value}
  };
  const response = await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const body = await response.json();
  if (!response.ok) { $('form-error').textContent = body.error; $('launch').disabled=false; return; }
  $('live').hidden=false; poll(body.run_id);
});
async function poll(runId) {
  const page = await fetch('/api/runs/'+runId).then(r => r.json()); render(page);
  if (page.status==='queued'||page.status==='running') setTimeout(()=>poll(runId),350);
  else $('launch').disabled=false;
}
function metricNumber(value) {
  const number=Number(value); return Number.isFinite(number)?number.toFixed(4):'—';
}
function promotionMarkup(round) {
  const promotion=round.promotion;
  if (!promotion) return '';
  const ranks=Object.fromEntries((promotion.ranking||[]).map(row=>[row.candidate_id,row]));
  const reports=(promotion.reports||[]).map(report=>{
    const rank=ranks[report.candidate_id]||{};
    const distributions=Object.entries(report.distributions||{}).map(([name,metric])=>
      `<tr><td>${escapeHtml(name.replaceAll('_',' '))}</td><td>${Number(metric.case_count)}</td><td>${metricNumber(metric.mean)}</td><td>${metricNumber(metric.cvar)}</td><td>${metricNumber(metric.maximum)}</td></tr>`).join('');
    return `<article class="promotion-report"><h3><span>${escapeHtml(report.candidate_id)}</span><small>${rank.rank?'Rank '+Number(rank.rank):'Not ranked'} · ${report.technically_eligible?'Eligible':'Not eligible'}</small></h3>
      <table><thead><tr><th>Distribution</th><th>Cases</th><th>Mean</th><th>CVaR</th><th>Max</th></tr></thead><tbody>${distributions}</tbody></table></article>`;
  }).join('');
  return `<section class="promotion-evidence"><h3>Round ${Number(round.round)} official hidden evaluation</h3>
    <p><strong>${promotion.promoted?'Candidate promoted':'Incumbent kept'}:</strong> ${escapeHtml(promotion.reason)}</p><div class="promotion-reports">${reports}</div></section>`;
}
function terminalRoundsMarkup(page) {
  const reports=page.terminal_round_reports||[];
  if (!reports.length) return '';
  const rows=reports.map(report=>{
    const distributions=Object.entries(report.distributions||{}).map(([name,metric])=>
      `${escapeHtml(name.replaceAll('_',' '))}: ${metricNumber(metric.mean)}`).join('<br>');
    const label=report.round===page.best_terminal_round?'Best diagnostic round':'';
    return `<tr><td>${Number(report.round)}</td><td>${escapeHtml(report.candidate_id)}</td><td>${distributions}</td><td>${label}</td></tr>`;
  }).join('');
  return `<section class="promotion-evidence"><h3>Terminal comparison across selected rounds</h3>
    <table><thead><tr><th>Round</th><th>Selected program</th><th>Terminal means</th><th>Diagnostic</th></tr></thead><tbody>${rows}</tbody></table></section>`;
}
function seedLedgerMarkup(page) {
  const entries=Object.entries(page.seed_ledger||{});
  if (!entries.length) return '';
  const rows=entries.map(([stage,seed])=>`<tr><td>${escapeHtml(stage)}</td><td>${Number(seed)}</td></tr>`).join('');
  return `<section class="promotion-evidence"><h3>Reproducibility seed ledger</h3>
    <p>Master seed ${Number(page.master_seed)}; ${entries.length} derived stage seeds.</p>
    <table><thead><tr><th>Stage</th><th>Seed</th></tr></thead><tbody>${rows}</tbody></table></section>`;
}
function evaluationResourceText(settings) {
  if (!settings || settings.mode === 'local_in_process') return 'Evaluation: local in-process backend';
  const device = settings.gpu_devices.length ? `${settings.gpu_devices.length} GPU, ${settings.gpu_slots_per_device} workers/GPU` : 'CPU';
  const placement = settings.development_solver_placement === 'agent_gpu'
    ? `Development on each Agent's own GPU (${Object.entries(settings.development_gpu_devices || {}).map(([agent, devices]) => `${agent}: ${devices.join(',')}`).join('; ')}); Official/Terminal: ${device}`
    : `Evaluation: ${device}`;
  return `${placement}; up to ${settings.solver_batch_size} Cases/solver batch; Oracle: ${settings.oracle_concurrency_per_device}/GPU, ${settings.oracle_max_workers} total; Development ${settings.development_oracle_max_workers ?? settings.oracle_max_workers}. Settings come from the launch configuration.`;
}
function submissionRuleText(policies) {
  const labels = {single_primary_v1:'one explicitly selected primary', latest:'last submitted version',
    all:'all submitted versions', program_defined:'local program returns its own result'};
  return Object.entries(policies || {}).map(([agent,policy]) => `${agent}: ${labels[policy] || policy}`).join('; ');
}
function render(page) {
  $('submission-rule').textContent = submissionRuleText(page.agent_submission_policies);
  $('evaluation-resources').textContent = evaluationResourceText(page.candidate_execution);
  $('status').textContent=page.activity || page.status;
  $('events').innerHTML=page.events.map(e=>`<div class="event"><small>${e.round?'Round '+Number(e.round):'Run'}</small><span>${escapeHtml(e.message)}</span></div>`).join('');
  $('events').scrollTop=$('events').scrollHeight;
  if (page.error) $('events').innerHTML += `<p class="error">${escapeHtml(page.error)}</p>`;
  const rounds=page.rounds||[];
  if (rounds.length) {
    const summary=`<table><thead><tr><th>Round</th><th>Selected</th>
      <th>Hidden error</th><th>Agent samples</th><th>Promotion</th></tr></thead><tbody>${rounds.map(r=>{
        const sampled=Object.entries(r.agent_case_counts||{}).map(([agent,counts])=>
          `${escapeHtml(agent)}: ${Object.values(counts).reduce((a,b)=>a+Number(b),0)}`).join('<br>');
        const selectedName=r.candidates.find(c=>c.artifact_id===r.selected_artifact_id)?.name||r.selected_artifact_id;
        return `<tr><td>${Number(r.round)}</td><td>${escapeHtml(selectedName)}</td>
          <td>${metricNumber(r.selected_error)} (${Number(r.hidden_suite_size)} Cases)</td><td>${sampled}</td><td>${escapeHtml(r.promotion?.reason||'')}</td></tr>`}).join('')}</tbody></table>`;
    $('rounds-view').innerHTML=summary+rounds.map(promotionMarkup).join('');
  } else $('rounds-view').innerHTML='Waiting for the first round…';
  $('memory').innerHTML=campaignMemoryMarkup(page.memory);
  $('candidate-history').innerHTML=candidateHistoryMarkup(page.candidate_history||[]);
  if (page.result) {
    const details = Object.entries(page.result.card.values).map(([name,value]) =>
      `<p><strong>${escapeHtml(name.replaceAll('_',' '))}:</strong> ${escapeHtml(value)}</p>`).join('');
    const terminal = Object.entries(page.result.terminal_distributions||{}).map(([name,row]) =>
      `<p><strong>${escapeHtml(name)} terminal mean:</strong> ${metricNumber(row.mean)}</p>`).join('');
    $('result').hidden=false;
    $('result').innerHTML=`<h2>Final Result</h2><h3>${escapeHtml(page.result.card.title)}</h3>
      <p>${escapeHtml(page.result.card.summary)}</p>
      <p><strong>Final score distribution:</strong> ${escapeHtml(page.result.terminal_score_distribution)}</p>
      ${details}${terminal}${terminalRoundsMarkup(page)}${seedLedgerMarkup(page)}
      <p><strong>Use:</strong> <code>${escapeHtml(page.result.usage)}</code></p>`;
  }
}
setup();
</script></main></body></html>"""
