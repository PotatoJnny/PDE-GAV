"""Product-facing browser panels for persistent PDE campaigns."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

from pde_agent_verifier_solver.panel.campaign_view import (
    build_campaign_view,
    build_dashboard_snapshot,
)


def run_campaign_dashboard(
    runs_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve a local fallback for the fixed PDE Agent web panel."""

    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"campaign runs folder does not exist: {root}")
    server = ThreadingHTTPServer((host, port), _dashboard_handler(root))
    print(f"PDE Agent panel: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def discover_staged_runs(runs_root: Path | str) -> list[Mapping[str, object]]:
    """Return compact summaries without paths, commands, job ids, or logs."""

    snapshot = build_dashboard_snapshot(runs_root)
    return [
        {
            "id": run["id"],
            "name": run["name"],
            "problem": run["problem"],
            "task": run["task"],
            "status": run["status"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "progress_percent": run["progress_percent"],
            "current_stage": run["current_phase"],
            "current_activity": run["current_activity"],
            "completed_stages": sum(
                phase["status"] == "completed" for phase in run["phases"]
            ),
            "total_stages": len(run["phases"]),
            "has_result": run["result"] is not None,
            "summary": run["summary"],
        }
        for run in snapshot["runs"]
    ]


def run_staged_panel(
    run_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    root = Path(run_root).resolve()
    if not (root / "run.json").is_file():
        raise FileNotFoundError(f"not a PDE campaign run: {root}")
    server = ThreadingHTTPServer((host, port), _single_handler(root))
    print(f"PDE Agent run: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _single_handler(run_root: Path):
    class Handler(_ProductHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"", "/"}:
                self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            elif path in {"/api/run", "/api/runs/" + run_root.name}:
                self._json(200, build_campaign_view(run_root))
            elif path.rstrip("/") == "/api/runs":
                run = build_campaign_view(run_root)
                self._json(
                    200,
                    {
                        "schema": "pde_panel_snapshot_v1",
                        "synced_at": run.get("updated_at"),
                        "counts": {
                            "running": int(run["status"] == "running"),
                            "submitted": int(run["status"] == "submitted"),
                            "completed": int(run["status"] == "completed"),
                            "failed": int(run["status"] == "failed"),
                        },
                        "runs": [run],
                    },
                )
            elif path == "/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

    return Handler


def _dashboard_handler(runs_root: Path):
    class Handler(_ProductHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"", "/"}:
                self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                return
            if path == "/health":
                self._json(200, {"status": "ok"})
                return
            if path.rstrip("/") in {"/api/runs", "/api/snapshot"}:
                self._json(200, build_dashboard_snapshot(runs_root))
                return
            prefix = "/api/runs/"
            if path.startswith(prefix):
                run_id = unquote(path[len(prefix) :])
                run_root = _find_run_root(runs_root, run_id)
                if run_root is None:
                    self._json(404, {"error": "campaign not found"})
                    return
                try:
                    run = build_campaign_view(run_root)
                except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    self._json(409, {"error": "campaign state is not ready"})
                    return
                self._json(200, run)
                return
            self._json(404, {"error": "not found"})

    return Handler


class _ProductHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: object) -> None:
        self._send(
            status,
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            ).encode(),
            "application/json; charset=utf-8",
        )

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)


def _run_directories(runs_root: Path) -> list[Path]:
    if (runs_root / "run.json").is_file():
        return [runs_root]
    if not runs_root.is_dir():
        return []
    return sorted(
        (
            child
            for child in runs_root.iterdir()
            if child.is_dir() and (child / "run.json").is_file()
        ),
        key=lambda path: path.name,
    )


def _find_run_root(runs_root: Path, run_id: str) -> Path | None:
    if not run_id or Path(run_id).name != run_id:
        return None
    return next((path for path in _run_directories(runs_root) if path.name == run_id), None)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PDE Agent</title><style>
:root{--ink:#17241f;--muted:#68746e;--paper:#fff;--bg:#f4f6f2;--line:#dde3dd;--green:#166746;--soft:#e5f1ea;--navy:#254b67;--orange:#c7772e;--red:#a64a42}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,system-ui,sans-serif}button{font:inherit}
header{height:68px;background:var(--paper);border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 28px;position:sticky;top:0;z-index:2}
.brand{display:flex;align-items:center;gap:11px;font-size:17px;font-weight:800}.mark{width:32px;height:32px;border-radius:9px;background:var(--ink);color:white;display:grid;place-items:center;font:700 19px Georgia,serif}
.sync{color:var(--muted);font-size:12px}.sync:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:#2b9a64;margin-right:9px;box-shadow:0 0 0 4px #e2f2e9}
main{max-width:1150px;margin:auto;padding:42px 25px 70px}h1{font:680 clamp(31px,4vw,44px)/1.05 Georgia,serif;letter-spacing:-.025em;margin:0}h2{font-size:18px;margin:0}h3{font-size:16px;margin:7px 0 4px}p{color:var(--muted)}.eyebrow{color:var(--green);font-size:10px;text-transform:uppercase;letter-spacing:.13em;font-weight:850;margin:0 0 7px}
.page-top,.section-top,.card,.hero,.metrics{display:flex;align-items:center;justify-content:space-between;gap:20px}.page-top{margin-bottom:34px}.section{margin-top:34px}.section-top{margin-bottom:13px}.quiet{color:var(--muted);font-size:11px}.count{background:var(--ink);color:white;border-radius:10px;padding:9px 13px}.count strong{font-size:21px;margin-right:6px}
.stack{display:grid;gap:11px}.card,.panel,.empty{width:100%;background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:19px 21px}.card{appearance:none;color:inherit;text-align:left;cursor:pointer;transition:.15s}.card:hover{transform:translateY(-2px);border-color:#aab7ae;box-shadow:0 9px 22px #1f342a0d}.card-main{display:flex;align-items:center;gap:15px;min-width:0}.card-main p{margin:0;font-size:12px}.icon{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;flex:none;font-weight:900}.completed .icon{color:var(--green);background:var(--soft)}.failed .icon{color:var(--red);background:#f6e8e5}.running .icon,.submitted .icon{color:var(--navy);background:#e5eef3}
.tags{display:flex;gap:6px}.tags span{padding:3px 7px;border-radius:5px;background:var(--soft);color:var(--green);font-size:9px;font-weight:850;text-transform:uppercase;letter-spacing:.06em}.tags span+span{background:#eef0ec;color:var(--muted)}
.card-metrics{display:flex;gap:25px;align-items:center}.card-metrics small,.metric small{display:block;color:var(--muted);font-size:10px}.card-metrics strong{font-size:15px}.positive{color:var(--green)}.progress-side{min-width:230px}.progress-line{height:7px;background:#e3e8e3;border-radius:99px;overflow:hidden;margin:7px 0}.progress-line i{display:block;height:100%;background:var(--navy)}.progress-side div:first-child{display:flex;justify-content:space-between}.progress-side small{color:var(--muted)}
.empty{display:flex;align-items:center;gap:15px}.empty-dot{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;background:var(--soft)}.empty-dot:after{content:"";width:9px;height:9px;border-radius:50%;background:var(--green)}.empty p{margin:3px 0 0}
.back{border:0;background:transparent;color:var(--green);font-weight:750;cursor:pointer;padding:0;margin-bottom:22px}.hero{align-items:flex-end;margin-bottom:27px}.hero p{max-width:720px;margin:11px 0 0}.pill{padding:7px 10px;border-radius:99px;background:var(--soft);color:var(--green);font-size:11px;font-weight:800}.issue{background:#fff7f5;border:1px solid #e5bbb6;border-radius:12px;padding:15px 17px;margin-bottom:20px}.issue strong{color:var(--red)}.issue p{margin:4px 0 0}
.metrics{align-items:stretch;margin:14px 0;display:grid;grid-template-columns:repeat(4,1fr);gap:11px}.metric{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:17px}.metric.primary{background:var(--green);color:white}.metric.primary small,.metric.primary span{color:#dcece3}.metric strong{display:block;font-size:25px;margin:8px 0 4px}.metric span{color:var(--muted);font-size:10px}
.panel{margin-top:13px}.phase-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:16px}.phase{border:1px solid var(--line);border-radius:10px;padding:12px;min-height:95px}.phase.completed{background:#f5faf6}.phase.running{border-color:#91afc0}.phase.failed{border-color:#d9aaa4}.phase b{display:block;font-size:11px}.phase small{color:var(--muted);font-size:9px}.phase-bar{height:4px;background:#e4e8e3;margin-top:16px}.phase-bar i{display:block;height:100%;background:var(--green)}
.rounds{margin-top:14px}.round{display:grid;grid-template-columns:50px 1fr repeat(3,110px);gap:12px;padding:13px 0;border-top:1px solid var(--line);align-items:center}.round:first-child{border-top:0}.round small{display:block;color:var(--muted);font-size:9px}.round strong{font-size:12px}
.promotion-round{padding:17px 0;border-top:1px solid var(--line)}.promotion-round:first-child{border-top:0}.promotion-head{display:flex;justify-content:space-between;gap:15px;margin-bottom:11px}.promotion-head p{margin:3px 0 0;font-size:11px}.promotion-state{color:var(--green);font-size:11px;font-weight:800}
.promotion-candidate{border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:8px}.promotion-candidate>div:first-child{display:flex;justify-content:space-between}.distribution-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:9px}.distribution{background:var(--bg);border-radius:8px;padding:9px}.distribution small,.distribution span{display:block;color:var(--muted);font-size:9px}.distribution strong{display:block;font-size:16px;margin:3px 0}
@media(max-width:780px){.card,.hero{align-items:flex-start;flex-direction:column}.card-metrics,.progress-side{width:100%;margin-left:51px}.metrics{grid-template-columns:repeat(2,1fr)}.phase-grid{grid-template-columns:repeat(2,1fr)}.round{grid-template-columns:40px 1fr}.round>div:nth-child(n+3){display:none}.distribution-grid{grid-template-columns:1fr}}
</style></head><body><header><div class="brand"><span class="mark">P</span>PDE Agent</div><span class="sync" id="sync">Connecting…</span></header><main id="app"><div class="empty">Loading campaigns…</div></main>
<script>
var app=document.querySelector("#app"),timer;
function esc(v){return String(v==null?"":v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;")}
function num(v){return v==null?"—":Number(v).toFixed(4)}
function pct(v){return v==null?"—":(Number(v)*100).toFixed(2)+"%"}
function get(path){return fetch(path,{cache:"no-store"}).then(function(r){return r.json().then(function(p){if(!r.ok)throw Error(p.error||"Request failed");return p})})}
function tags(run){return '<div class="tags"><span>'+esc(run.problem)+'</span><span>'+esc(run.status==="failed"?"Needs attention":run.status)+'</span></div>'}
function card(run){
 var right=run.status==="completed"?'<div class="card-metrics"><div><small>Hidden error</small><strong>'+num(run.summary.hidden_error)+'</strong></div><div><small>Overall gain</small><strong class="positive">'+pct(run.summary.improvement)+'</strong></div><b>→</b></div>':'<div class="progress-side"><div><strong>'+esc(run.progress_percent)+'%</strong><small>'+esc(run.current_phase)+'</small></div><div class="progress-line"><i style="width:'+esc(run.progress_percent)+'%"></i></div><small>'+esc(run.current_activity)+'</small></div>';
 if(run.status==="failed")right='<div class="progress-side"><strong>'+esc(run.issue&&run.issue.phase||run.current_phase)+'</strong><small>'+esc(run.issue&&run.issue.description||"Campaign stopped")+'</small></div>';
 return '<button class="card '+esc(run.status)+'" onclick="openRun(\''+encodeURIComponent(run.id)+'\')"><div class="card-main"><span class="icon">'+(run.status==="completed"?"✓":run.status==="failed"?"!":"●")+'</span><div>'+tags(run)+'<h3>'+esc(run.name)+'</h3><p>'+esc(run.current_activity)+'</p></div></div>'+right+'</button>'
}
function home(p){
 document.querySelector("#sync").textContent="Synced just now";
 var active=p.runs.filter(function(r){return r.status==="running"||r.status==="submitted"}),done=p.runs.filter(function(r){return r.status==="completed"}),failed=p.runs.filter(function(r){return r.status==="failed"});
 app.innerHTML='<div class="page-top"><div><p class="eyebrow">Experiment workspace</p><h1>Your PDE campaigns</h1><p>Follow experiments while they run, then inspect the final scientific result.</p></div><div class="count"><strong>'+done.length+'</strong>result ready</div></div>'+
 '<section class="section"><div class="section-top"><h2>Running now</h2><span class="quiet">Updates automatically</span></div>'+(active.length?'<div class="stack">'+active.map(card).join("")+'</div>':'<div class="empty"><span class="empty-dot"></span><div><strong>No campaign is running</strong><p>The next experiment will appear here as soon as it starts.</p></div></div>')+'</section>'+
 '<section class="section"><div class="section-top"><h2>Recent results</h2><span class="quiet">'+done.length+' completed</span></div><div class="stack">'+done.map(card).join("")+'</div></section>'+
 (failed.length?'<section class="section"><div class="section-top"><h2>Needs attention</h2><span class="quiet">'+failed.length+'</span></div><div class="stack">'+failed.map(card).join("")+'</div></section>':'');
 timer=setTimeout(loadHome,5000)
}
function phase(row){return '<div class="phase '+esc(row.status)+'"><b>'+esc(row.name)+'</b><small>'+esc(row.detail||row.status)+'</small><div class="phase-bar"><i style="width:'+Math.round(row.progress*100)+'%"></i></div></div>'}
function round(row){return '<div class="round"><strong>Round '+row.round+'</strong><div><small>What happened</small><strong>'+(row.improvement>0?"Solver improved":row.guided_update?"Guided challenge complete":"Solver evaluated")+'</strong></div><div><small>Current</small><strong>'+num(row.current&&row.current.mean_relative_l2)+'</strong></div><div><small>Previous</small><strong>'+num(row.previous&&row.previous.mean_relative_l2)+'</strong></div><div><small>Hidden</small><strong>'+num(row.hidden&&row.hidden.mean_relative_l2)+'</strong></div></div>'}
function promotionReport(report){var metrics=report.distributions.map(function(d){return '<div class="distribution"><small>'+esc(d.name.replaceAll("_"," "))+' · '+esc(d.case_count)+' cases</small><strong>'+num(d.mean)+'</strong><span>mean · CVaR '+num(d.cvar)+' · max '+num(d.max)+'</span></div>'}).join("");return '<div class="promotion-candidate"><div><strong>'+esc(report.candidate)+'</strong><span class="quiet">'+(report.technically_eligible?"Eligible":"Not eligible")+'</span></div><div class="distribution-grid">'+metrics+'</div></div>'}
function promotionRound(row){return '<div class="promotion-round"><div class="promotion-head"><div><strong>Round '+esc(row.round)+' · '+esc(row.selected)+'</strong><p>'+esc(row.reason||"Selection rule completed")+'</p></div><span class="promotion-state">'+(row.promoted?"Promoted":"Incumbent kept")+'</span></div>'+row.reports.map(promotionReport).join("")+'</div>'}
function detail(run){
 clearTimeout(timer);document.title=run.name+" · PDE Agent";
 var issue=run.issue?'<div class="issue"><strong>'+esc(run.issue.title)+'</strong><p>'+esc(run.issue.description)+'</p></div>':'';
 var result=run.result,latest=run.live.latest_metrics,rounds=result?result.rounds:[],promotions=result&&result.promotion_history;
 app.innerHTML='<button class="back" onclick="back()">← All runs</button>'+issue+'<div class="hero"><div>'+tags(run)+'<h1>'+esc(run.name)+'</h1><p>'+esc(run.status==="completed"?"The final "+run.task.toLowerCase()+" is ready to inspect and use.":run.current_activity)+'</p></div><span class="pill">'+esc(run.progress_percent)+'% · '+esc(run.current_phase)+'</span></div>'+
 '<div class="metrics"><div class="metric primary"><small>'+(result?"Hidden-test relative L2":"Current solver error")+'</small><strong>'+num(result&&result.primary_error!=null?result.primary_error:latest.solver_error)+'</strong><span>Lower is better</span></div><div class="metric"><small>Previous solver</small><strong>'+num(latest.previous_error)+'</strong><span>Same challenge cases</span></div><div class="metric"><small>Generated cases</small><strong>'+esc(run.summary.evaluated_cases)+'</strong><span>Across finished rounds</span></div><div class="metric"><small>Rounds</small><strong>'+esc(run.rounds_completed)+' / '+esc(run.rounds_requested)+'</strong><span>Completed</span></div></div>'+
 '<section class="panel"><div class="section-top"><div><p class="eyebrow">Campaign progress</p><h2>From data to final result</h2></div></div><div class="phase-grid">'+run.phases.map(phase).join("")+'</div></section>'+
 (rounds.length?'<section class="panel"><div class="section-top"><div><p class="eyebrow">Solver performance</p><h2>Error across rounds</h2></div><span class="quiet">Lower is better</span></div><div class="rounds">'+rounds.map(round).join("")+'</div></section>':'')+
 (promotions&&promotions.rounds.length?'<section class="panel"><div class="section-top"><div><p class="eyebrow">Candidate selection</p><h2>Why this program entered the next round</h2></div><span class="quiet">'+esc(promotions.program||"Configured rule")+'</span></div>'+promotions.rounds.map(promotionRound).join("")+'</section>':'');
 if(run.status==="running"||run.status==="submitted")timer=setTimeout(function(){openRun(encodeURIComponent(run.id),true)},3000)
}
function loadHome(){get("api/runs").then(home).catch(function(e){app.innerHTML='<div class="issue"><strong>Panel unavailable</strong><p>'+esc(e.message)+'</p></div>';timer=setTimeout(loadHome,5000)})}
function openRun(id,silent){get("api/runs/"+id).then(detail).catch(function(e){if(!silent)alert(e.message)})}
function back(){history.replaceState(null,"",location.pathname);document.title="PDE Agent";loadHome()}
window.openRun=function(id){history.pushState(null,"","?run="+id);openRun(id)};window.back=back;
var initial=new URLSearchParams(location.search).get("run");if(initial)openRun(encodeURIComponent(initial));else loadHome();
</script></body></html>"""

HTML = DASHBOARD_HTML
