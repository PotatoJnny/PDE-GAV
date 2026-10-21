"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Status = "running" | "submitted" | "completed" | "failed";

interface MetricSummary {
  mean_relative_l2?: number;
  median_relative_l2?: number;
  q10_relative_l2?: number;
  q90_relative_l2?: number;
  min_relative_l2?: number;
  max_relative_l2?: number;
  n?: number;
  ok?: number;
}

interface RoundView {
  round: number;
  status: string;
  generated_cases?: number;
  replay_cases?: number;
  feedback_cases?: number;
  current?: MetricSummary;
  previous?: MetricSummary;
  hidden?: MetricSummary;
  cumulative?: MetricSummary;
  improvement?: number;
  cumulative_improvement?: number;
  physics?: { accepted: number; total: number };
  guided_update?: boolean;
}

interface GeneratorView {
  basis_rank?: number;
  selected_checkpoint_step?: number;
  basis: Array<{ rank?: number; explained_variance?: number }>;
  train_cases?: number;
  validation_cases?: number;
  training_step?: number;
  training_steps?: number;
  validation_loss?: number;
  loss_history: Array<{ step: number; train_loss?: number; validation_loss?: number }>;
}

interface CampaignResult {
  kind: string;
  name: string;
  primary_error?: number;
  overall_improvement?: number;
  rounds: RoundView[];
  hidden_distribution: Array<{ case: string; error: number }>;
  generated_distribution: Array<{ case: string; error: number }>;
  physics: { accepted: number; total: number };
  wall_time_seconds?: number;
  generator: GeneratorView;
  memory: { rounds_summarized: number; insights: string[] };
  artifact: { available: boolean; available_on_cluster?: boolean; filename?: string; source?: string; usage?: string };
}

interface CampaignRun {
  id: string;
  name: string;
  problem: string;
  problem_id: string;
  task: string;
  status: Status;
  created_at?: string;
  updated_at?: string;
  finished_at?: string;
  elapsed_seconds?: number;
  progress: number;
  progress_percent: number;
  current_phase: string;
  current_activity: string;
  rounds_completed: number;
  rounds_requested: number;
  cases_per_round?: number;
  phases: Array<{ id: string; name: string; status: Status; progress: number; detail?: string }>;
  live: {
    current_round: number;
    rounds_requested: number;
    activity: string;
    latest_metrics: {
      solver_error?: number;
      previous_error?: number;
      hidden_error?: number;
    };
    generator: GeneratorView;
    case_mix: {
      original?: number;
      generated?: number;
      base_or_guided?: number;
      perturbed?: number;
      replay?: number;
      guided?: boolean;
    };
    hidden_suite?: {
      fixed_reference?: number;
      new_per_round?: number;
      accumulate?: boolean;
    };
    recent_events: Array<{ round: number; title: string; metric?: number }>;
  };
  summary: {
    primary_error?: number;
    hidden_error?: number;
    generated_error?: number;
    improvement?: number;
    evaluated_cases: number;
  };
  result?: CampaignResult;
  issue?: { title: string; description: string; phase?: string };
}

interface Snapshot {
  schema: string;
  synced_at: string;
  counts: Record<Status, number>;
  runs: CampaignRun[];
}

const emptySnapshot: Snapshot = {
  schema: "pde_panel_snapshot_v1",
  synced_at: "",
  counts: { running: 0, submitted: 0, completed: 0, failed: 0 },
  runs: [],
};

const colors = {
  current: "#166746",
  previous: "#c7772e",
  hidden: "#254b67",
  grid: "#e2e7e1",
  text: "#6a756f",
};

function number(value: number | undefined, digits = 4) {
  if (value === undefined || value === null || !Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(digits);
}

function percent(value: number | undefined, digits = 2) {
  if (value === undefined || value === null || !Number.isFinite(value)) return "—";
  return (value * 100).toFixed(digits) + "%";
}

function duration(seconds: number | undefined) {
  if (seconds === undefined || !Number.isFinite(seconds)) return "—";
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours) return String(hours) + "h " + String(minutes) + "m";
  if (minutes) return String(minutes) + "m " + String(secs) + "s";
  return String(secs) + "s";
}

function relativeTime(value: string | undefined) {
  if (!value) return "Not synced yet";
  const seconds = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 10) return "Synced just now";
  if (seconds < 60) return "Synced " + String(seconds) + "s ago";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return "Synced " + String(minutes) + "m ago";
  return "Last synced " + new Date(value).toLocaleString();
}

function dateLabel(value: string | undefined) {
  if (!value) return "—";
  const date = new Date(value);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function campaignLabel(status: Status) {
  return {
    running: "Running",
    submitted: "Starting",
    completed: "Completed",
    failed: "Needs attention",
  }[status];
}

function LineChart({
  labels,
  series,
  height = 250,
}: {
  labels: string[];
  series: Array<{ label: string; color: string; values: Array<number | undefined> }>;
  height?: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const draw = () => {
      const width = Math.max(element.clientWidth, 320);
      const ratio = window.devicePixelRatio || 1;
      element.width = width * ratio;
      element.height = height * ratio;
      const context = element.getContext("2d");
      if (!context) return;
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);
      const left = 54;
      const right = 18;
      const top = 20;
      const bottom = 38;
      const plotWidth = width - left - right;
      const plotHeight = height - top - bottom;
      const values = series.flatMap((row) => row.values).filter((value): value is number => value !== undefined && Number.isFinite(value));
      if (!values.length) return;
      let min = Math.min(...values);
      let max = Math.max(...values);
      const padding = Math.max((max - min) * 0.18, Math.abs(max || 1) * 0.003);
      min -= padding;
      max += padding;
      const x = (index: number) => labels.length <= 1 ? left + plotWidth / 2 : left + (index / (labels.length - 1)) * plotWidth;
      const y = (value: number) => top + (1 - (value - min) / (max - min || 1)) * plotHeight;

      context.lineWidth = 1;
      context.font = "11px Inter, system-ui, sans-serif";
      context.fillStyle = colors.text;
      for (let index = 0; index < 4; index += 1) {
        const yy = top + (index / 3) * plotHeight;
        const value = max - (index / 3) * (max - min);
        context.strokeStyle = colors.grid;
        context.beginPath();
        context.moveTo(left, yy);
        context.lineTo(width - right, yy);
        context.stroke();
        context.textAlign = "right";
        context.fillText(value.toFixed(3), left - 10, yy + 4);
      }

      labels.forEach((label, index) => {
        context.textAlign = "center";
        context.fillText(label, x(index), height - 12);
      });

      series.forEach((row) => {
        context.strokeStyle = row.color;
        context.fillStyle = row.color;
        context.lineWidth = 2.5;
        context.beginPath();
        let started = false;
        row.values.forEach((value, index) => {
          if (value === undefined) return;
          if (!started) {
            context.moveTo(x(index), y(value));
            started = true;
          } else {
            context.lineTo(x(index), y(value));
          }
        });
        context.stroke();
        row.values.forEach((value, index) => {
          if (value === undefined) return;
          context.beginPath();
          context.arc(x(index), y(value), 4, 0, Math.PI * 2);
          context.fill();
          context.strokeStyle = "#ffffff";
          context.lineWidth = 2;
          context.stroke();
        });
      });
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(element);
    return () => observer.disconnect();
  }, [height, labels, series]);

  return (
    <div className="chart">
      <canvas ref={canvas} style={{ height }} aria-label="Error across campaign rounds" />
      <div className="legend">
        {series.map((row) => <span key={row.label}><i style={{ background: row.color }} />{row.label}</span>)}
      </div>
    </div>
  );
}

function SolverChart({ rounds }: { rounds: RoundView[] }) {
  const labels = rounds.map((row) => "Round " + String(row.round));
  const series = useMemo(() => [
    { label: "Current solver", color: colors.current, values: rounds.map((row) => row.current?.mean_relative_l2) },
    { label: "Previous solver", color: colors.previous, values: rounds.map((row) => row.previous?.mean_relative_l2) },
    { label: "Hidden test", color: colors.hidden, values: rounds.map((row) => row.hidden?.mean_relative_l2) },
  ], [rounds]);
  return <LineChart labels={labels} series={series} />;
}

function GeneratorLossChart({ generator }: { generator: GeneratorView }) {
  const rows = generator.loss_history;
  const series = useMemo(() => [
    { label: "Training loss", color: colors.current, values: rows.map((row) => row.train_loss) },
    { label: "Validation loss", color: colors.hidden, values: rows.map((row) => row.validation_loss) },
  ], [rows]);
  if (!rows.length) return <p className="empty-copy">Loss history will appear when Generator training starts.</p>;
  return <LineChart labels={rows.map((row) => String(row.step))} series={series} height={220} />;
}

function PhaseStepper({ run }: { run: CampaignRun }) {
  return (
    <div className="phase-stepper">
      {run.phases.map((phase, index) => (
        <div className={"phase-step " + phase.status} key={phase.id}>
          <span className="phase-index">{phase.status === "completed" ? "✓" : String(index + 1)}</span>
          <div><strong>{phase.name}</strong><small>{phase.detail || campaignLabel(phase.status)}</small></div>
          <div className="mini-progress"><i style={{ width: String(Math.round(phase.progress * 100)) + "%" }} /></div>
        </div>
      ))}
    </div>
  );
}

function RunCard({ run, onOpen }: { run: CampaignRun; onOpen: () => void }) {
  const active = run.status === "running" || run.status === "submitted";
  const completed = run.status === "completed";
  return (
    <button className={"run-card " + run.status} onClick={onOpen}>
      <div className="run-card-main">
        <span className={"status-icon " + run.status}>{completed ? "✓" : active ? "●" : "!"}</span>
        <div>
          <div className="run-label"><span>{run.problem}</span><span>{campaignLabel(run.status)}</span></div>
          <h3>{run.name}</h3>
          <p>{run.status === "failed" ? run.issue?.description : run.current_activity}</p>
        </div>
      </div>
      {active ? (
        <div className="run-progress-summary">
          <div><strong>{run.progress_percent}%</strong><small>{run.current_phase}</small></div>
          <div className="progress-track"><i style={{ width: String(run.progress_percent) + "%" }} /></div>
          <span>Round {run.rounds_completed + 1} / {run.rounds_requested}</span>
        </div>
      ) : completed ? (
        <div className="run-metrics">
          <div><small>Hidden error</small><strong>{number(run.summary.hidden_error)}</strong></div>
          <div><small>Overall gain</small><strong className="positive">{percent(run.summary.improvement)}</strong></div>
          <span className="arrow">→</span>
        </div>
      ) : (
        <div className="failure-meta"><small>Stopped in</small><strong>{run.issue?.phase || run.current_phase}</strong><span>{dateLabel(run.updated_at)}</span></div>
      )}
    </button>
  );
}

function MetricCard({ label, value, detail, primary = false }: { label: string; value: string; detail: string; primary?: boolean }) {
  return <article className={"metric-card" + (primary ? " primary" : "")}><small>{label}</small><strong>{value}</strong><span>{detail}</span></article>;
}

function CaseMix({ run }: { run: CampaignRun }) {
  const mix = run.live.case_mix;
  const original = mix.original || 0;
  const generated = mix.generated ?? mix.base_or_guided ?? 0;
  const perturbed = mix.perturbed || 0;
  const total = original + generated + perturbed || 1;
  const hidden = run.live.hidden_suite;
  return (
    <div className="case-mix">
      <div className="mix-bar"><i className="mix-original" style={{ width: String((original / total) * 100) + "%" }} /><i className="mix-base" style={{ width: String((generated / total) * 100) + "%" }} /><i className="mix-perturbed" style={{ width: String((perturbed / total) * 100) + "%" }} /></div>
      <div className="mix-legend">
        <span><i className="mix-original" />Original D0 <b>{original || "—"}</b></span>
        <span><i className="mix-base" />Generated{mix.guided ? " (base + guided)" : ""} <b>{generated || "—"}</b></span>
        <span><i className="mix-perturbed" />Perturbed <b>{perturbed || "—"}</b></span>
        <span><i className="mix-replay" />Replay from earlier rounds <b>{mix.replay || 0}</b></span>
      </div>
      {hidden ? <p className="hidden-suite-note">Hidden suite: {hidden.fixed_reference || 0} fixed reference + {hidden.new_per_round || 0} new cases per round{hidden.accumulate ? ", accumulated across rounds" : ""}.</p> : null}
    </div>
  );
}

function BasisQuality({ generator }: { generator: GeneratorView }) {
  if (!generator.basis.length) return <p className="empty-copy">Basis quality will appear after coefficient projection.</p>;
  return (
    <div className="basis-list">
      {generator.basis.map((row) => (
        <div className="basis-row" key={row.rank}>
          <span>Rank {row.rank}</span>
          <div className="basis-track"><i style={{ width: String((row.explained_variance || 0) * 100) + "%" }} /></div>
          <strong>{percent(row.explained_variance, 1)}</strong>
        </div>
      ))}
    </div>
  );
}

function ErrorDistribution({ values }: { values: Array<{ case: string; error: number }> }) {
  if (!values.length) return <p className="empty-copy">Case-level errors are not available yet.</p>;
  const errors = values.map((row) => row.error);
  let min = Math.min(...errors);
  let max = Math.max(...errors);
  const pad = Math.max((max - min) * 0.08, Math.abs(max || 1) * 0.0005);
  min -= pad;
  max += pad;
  return (
    <div className="distribution">
      <div className="distribution-axis">
        {values.map((row, index) => {
          const left = ((row.error - min) / (max - min || 1)) * 100;
          return <button key={row.case + String(index)} className="case-dot" style={{ left: String(left) + "%", top: String(20 + (index % 4) * 15) + "px" }} title={row.case + ": " + number(row.error, 6)} aria-label={row.case + " error " + number(row.error, 6)} />;
        })}
      </div>
      <div className="distribution-labels"><span>{number(min, 4)}</span><span>Relative L2 error</span><span>{number(max, 4)}</span></div>
    </div>
  );
}

function LiveRunPage({ run, onBack }: { run: CampaignRun; onBack: () => void }) {
  const latest = run.live.latest_metrics;
  const rounds = run.result?.rounds || [];
  const isFailed = run.status === "failed";
  return (
    <section className="content detail-view">
      <button className="back-button" onClick={onBack}>← All runs</button>
      {isFailed && run.issue ? <div className="issue-banner"><span>!</span><div><strong>{run.issue.title}</strong><p>{run.issue.description}</p></div></div> : null}
      <div className="live-hero">
        <div>
          <div className="run-label"><span>{run.problem}</span><span>{run.task}</span></div>
          <h1>{run.name}</h1>
          <p>{isFailed ? "Intermediate results are preserved below." : run.current_activity}</p>
        </div>
        <div className={"progress-ring " + run.status} style={{ background: "conic-gradient(var(--green) " + String(run.progress_percent) + "%, #e3e8e3 0)" }}>
          <div><strong>{run.progress_percent}%</strong><span>{campaignLabel(run.status)}</span></div>
        </div>
      </div>

      <div className="live-strip">
        <div><small>Current phase</small><strong>{run.current_phase}</strong></div>
        <div><small>Campaign round</small><strong>{Math.min(run.live.current_round, run.rounds_requested)} of {run.rounds_requested}</strong></div>
        <div><small>Elapsed</small><strong>{duration(run.elapsed_seconds)}</strong></div>
        <div><small>Cases per round</small><strong>{run.cases_per_round || "—"}</strong></div>
      </div>

      <section className="panel phase-panel"><div className="panel-heading"><div><p className="eyebrow">Campaign progress</p><h2>From data to final solver</h2></div><span className={run.status === "running" ? "live-label" : "quiet"}>{campaignLabel(run.status)}</span></div><PhaseStepper run={run} /></section>

      <div className="metric-grid three">
        <MetricCard label="Current solver error" value={number(latest.solver_error)} detail="On the latest generated cases" primary />
        <MetricCard label="Previous solver error" value={number(latest.previous_error)} detail="Same round, before Agent work" />
        <MetricCard label="Hidden-test error" value={number(latest.hidden_error)} detail="Independent cases kept from the Agent" />
      </div>

      {rounds.length ? <section className="panel chart-panel"><div className="panel-heading"><div><p className="eyebrow">Intermediate solver</p><h2>Error across finished rounds</h2></div><span className="quiet">Lower is better</span></div><SolverChart rounds={rounds} /></section> : null}

      <div className="two-column">
        <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Generator training</p><h2>Learning progress</h2></div><span className="quiet">{run.live.generator.selected_checkpoint_step ? <>Selected checkpoint {run.live.generator.selected_checkpoint_step.toLocaleString()}</> : <>Step {run.live.generator.training_step || 0} / {run.live.generator.training_steps || "—"}</>}</span></div><GeneratorLossChart generator={run.live.generator} /></section>
        <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Round curriculum</p><h2>Cases shown to the Agent</h2></div></div><CaseMix run={run} /><div className="subsection"><small className="subheading">Coefficient basis coverage</small><BasisQuality generator={run.live.generator} /></div></section>
      </div>

      {run.live.recent_events.length ? <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Recent progress</p><h2>What the campaign completed</h2></div></div><div className="activity-list">{run.live.recent_events.map((event) => <div key={event.round}><span>✓</span><strong>{event.title}</strong></div>)}</div></section> : null}
    </section>
  );
}

function downloadSolver(result: CampaignResult) {
  if (!result.artifact.source) return;
  const blob = new Blob([result.artifact.source], { type: "text/x-python;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = result.artifact.filename || "submission.py";
  link.click();
  URL.revokeObjectURL(url);
}

function ResultPage({ run, onBack }: { run: CampaignRun; onBack: () => void }) {
  const result = run.result;
  if (!result) return <LiveRunPage run={run} onBack={onBack} />;
  const finalRound = result.rounds[result.rounds.length - 1];
  const hidden = finalRound?.hidden;
  return (
    <section className="content detail-view">
      <button className="back-button" onClick={onBack}>← All results</button>
      <div className="result-hero">
        <div>
          <div className="run-label"><span>{run.problem}</span><span className="complete-label">Result ready</span></div>
          <h1>{run.name}</h1>
          <p>The final {run.task.toLowerCase()} completed {run.rounds_completed} Agent–Generator rounds and is ready to use.</p>
        </div>
        {result.artifact.available ? (
          <button className="primary-button" onClick={() => downloadSolver(result)}>↓ Download final solver</button>
        ) : result.artifact.available_on_cluster ? (
          <span className="solver-saved">✓ Final solver saved on the cluster</span>
        ) : null}
      </div>

      <div className="metric-grid">
        <MetricCard label="Hidden-test relative L2" value={number(result.primary_error)} detail={hidden?.q10_relative_l2 !== undefined ? number(hidden.q10_relative_l2) + "–" + number(hidden.q90_relative_l2) + " middle 80%" : "Independent evaluation"} primary />
        <MetricCard label="Generated-case error" value={number(finalRound?.cumulative?.mean_relative_l2)} detail={String(finalRound?.cumulative?.n || run.summary.evaluated_cases) + " cumulative cases"} />
        <MetricCard label="Physics checks" value={String(result.physics.accepted) + " / " + String(result.physics.total)} detail="Continuous contracts passed" />
        <MetricCard label="Campaign runtime" value={duration(result.wall_time_seconds)} detail={String(result.rounds.length) + " rounds completed"} />
      </div>

      <div className="result-grid">
        <section className="panel chart-panel"><div className="panel-heading"><div><p className="eyebrow">Solver performance</p><h2>Error across rounds</h2></div><span className="quiet">Lower is better</span></div><SolverChart rounds={result.rounds} /></section>
        <section className="panel round-history"><div className="panel-heading"><div><p className="eyebrow">Round history</p><h2>What changed</h2></div></div>
          {result.rounds.map((row) => {
            const gain = row.improvement;
            const stable = gain === undefined || Math.abs(gain) < 0.00005;
            return <div className="round-row" key={row.round}><span>{String(row.round).padStart(2, "0")}</span><div><strong>{stable ? "Solver stayed stable" : gain > 0 ? "Solver improved on new cases" : "Generator found a harder region"}</strong><p>{row.generated_cases || 0} new cases · {row.replay_cases || 0} replay cases · {row.physics?.accepted || 0}/{row.physics?.total || 0} physics checks</p></div><b className={gain && gain > 0 ? "positive" : ""}>{stable ? "Stable" : percent(gain)}</b></div>;
          })}
        </section>
      </div>

      <div className="two-column">
        <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Case-level evaluation</p><h2>Generated error distribution</h2></div><span className="quiet">{result.generated_distribution.length} cases</span></div><ErrorDistribution values={result.generated_distribution} /></section>
        <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Independent evaluation</p><h2>Hidden-test cases</h2></div><span className="quiet">{result.hidden_distribution.length} cases</span></div><ErrorDistribution values={result.hidden_distribution} /></section>
      </div>

      <div className="two-column">
        <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Generator representation</p><h2>Basis quality</h2></div><span className="quiet">Selected rank {result.generator.basis_rank || "—"}</span></div><BasisQuality generator={result.generator} /></section>
        <section className="panel memory-panel"><div className="panel-heading"><div><p className="eyebrow">Campaign memory</p><h2>What the Agent keeps</h2></div><span className="quiet">{result.memory.rounds_summarized} rounds</span></div>
          {result.memory.insights.length ? <ul>{result.memory.insights.map((insight) => <li key={insight}>{insight}</li>)}</ul> : <p className="empty-copy">The memory contains round metrics; no additional scientific insight was recorded.</p>}
        </section>
      </div>
    </section>
  );
}

function Overview({
  snapshot,
  section,
  openRun,
}: {
  snapshot: Snapshot;
  section: "runs" | "results";
  openRun: (run: CampaignRun) => void;
}) {
  const running = snapshot.runs.filter((run) => run.status === "running" || run.status === "submitted");
  const completed = snapshot.runs.filter((run) => run.status === "completed");
  const failed = snapshot.runs.filter((run) => run.status === "failed");
  if (section === "results") {
    return (
      <section className="content">
        <div className="page-heading"><div><p className="eyebrow">Scientific outputs</p><h1>Results</h1><p className="heading-copy">Final solvers and models, with the evidence needed to understand their performance.</p></div><div className="summary-chip"><strong>{completed.length}</strong><span>ready to inspect</span></div></div>
        <section className="section-block"><div className="section-heading"><h2>Completed campaigns</h2><span className="quiet">Newest first</span></div><div className="card-stack">{completed.map((run) => <RunCard key={run.id} run={run} onOpen={() => openRun(run)} />)}</div></section>
      </section>
    );
  }
  return (
    <section className="content">
      <div className="page-heading"><div><p className="eyebrow">Experiment workspace</p><h1>Your PDE campaigns</h1><p className="heading-copy">Follow experiments while they run, then inspect the final scientific result.</p></div><div className="summary-chip"><strong>{completed.length}</strong><span>result ready</span></div></div>

      <section className="section-block">
        <div className="section-heading"><h2>Running now</h2><span className="live-label"><i />Updates automatically</span></div>
        {running.length ? <div className="card-stack">{running.map((run) => <RunCard key={run.id} run={run} onOpen={() => openRun(run)} />)}</div> : <div className="empty-live"><div className="pulse-ring"><span /></div><div><strong>No campaign is running</strong><p>The next experiment will appear here as soon as it starts.</p></div></div>}
      </section>

      <section className="section-block">
        <div className="section-heading"><h2>Recent results</h2><span className="quiet">{completed.length} completed</span></div>
        <div className="card-stack">{completed.slice(0, 4).map((run) => <RunCard key={run.id} run={run} onOpen={() => openRun(run)} />)}</div>
      </section>

      {failed.length ? <section className="section-block attention"><div className="section-heading"><h2>Needs attention</h2><span className="count-badge">{failed.length}</span></div><div className="card-stack compact">{failed.map((run) => <RunCard key={run.id} run={run} onOpen={() => openRun(run)} />)}</div></section> : null}
    </section>
  );
}

export default function Dashboard({
  viewerName,
  signOutPath,
}: {
  viewerName: string;
  signOutPath: string;
}) {
  const [snapshot, setSnapshot] = useState<Snapshot>(emptySnapshot);
  const [section, setSection] = useState<"runs" | "results">("runs");
  const [selectedId, setSelectedId] = useState<string | null>(() =>
    typeof window === "undefined"
      ? null
      : new URLSearchParams(window.location.search).get("run"),
  );
  const [, tick] = useState(0);

  useEffect(() => {
    const onPop = () => setSelectedId(new URLSearchParams(window.location.search).get("run"));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      try {
        const response = await fetch("/api/snapshot", { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json() as Snapshot;
        if (!cancelled && payload.schema === "pde_panel_snapshot_v1") setSnapshot(payload);
      } catch {
        // Keep the last good snapshot visible during a transient sync issue.
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 5000);
    const clock = window.setInterval(() => tick((value) => value + 1), 10000);
    return () => { cancelled = true; window.clearInterval(timer); window.clearInterval(clock); };
  }, []);

  const selected = snapshot.runs.find((run) => run.id === selectedId);
  const openRun = (run: CampaignRun) => {
    setSelectedId(run.id);
    window.history.pushState(null, "", "?run=" + encodeURIComponent(run.id));
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const closeRun = () => {
    setSelectedId(null);
    window.history.pushState(null, "", window.location.pathname);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <main>
      <header className="app-header">
        <button className="brand" onClick={() => { closeRun(); setSection("runs"); }} aria-label="PDE Agent home"><span className="brand-mark">P</span><span>PDE Agent</span></button>
        <div className="header-meta">
          <span className="sync-dot" />{relativeTime(snapshot.synced_at)}
          <span className="viewer-name">{viewerName}</span>
          {signOutPath ? <a className="sign-out" href={signOutPath}>Sign out</a> : null}
        </div>
      </header>
      <div className="workspace">
        <nav className="sidebar" aria-label="Main navigation">
          <button className={!selected && section === "runs" ? "nav-active" : ""} onClick={() => { closeRun(); setSection("runs"); }}><span>Runs</span><b>{snapshot.runs.length}</b></button>
          <button className={!selected && section === "results" ? "nav-active" : ""} onClick={() => { closeRun(); setSection("results"); }}><span>Results</span><b>{snapshot.counts.completed || 0}</b></button>
          <div className="sidebar-note"><span>Workspace</span><strong>{snapshot.counts.running ? String(snapshot.counts.running) + " running" : "Ready"}</strong><small>{snapshot.counts.completed || 0} scientific results</small></div>
        </nav>
        {selected ? selected.status === "completed" && selected.result ? <ResultPage run={selected} onBack={closeRun} /> : <LiveRunPage run={selected} onBack={closeRun} /> : <Overview snapshot={snapshot} section={section} openRun={openRun} />}
      </div>
    </main>
  );
}
