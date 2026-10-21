"""Panel checks for the chronological Campaign memory presentation."""

from pde_agent_verifier_solver.panel.server import HTML


def test_panel_presents_campaign_memory_separately_from_candidate_history() -> None:
    assert "<h2>Campaign memory</h2>" in HTML
    assert "<h2>Candidate history</h2>" in HTML
    assert "<h2>Memory tree</h2>" not in HTML
    assert "campaignMemoryMarkup(page.memory)" in HTML
    assert "candidateHistoryMarkup(page.candidate_history||[])" in HTML


def test_agent_authored_memory_is_escaped_before_browser_rendering() -> None:
    assert "escapeHtml(log.markdown)" in HTML
    assert "escapeHtml(round.controller_markdown)" in HTML
    assert "escapeHtml(name)" in HTML
    assert "escapeHtml(value)" in HTML
    assert "escapeHtml(e.message)" in HTML
    assert "escapeHtml(page.error)" in HTML
    assert "node.observations" not in HTML


def test_panel_escapes_candidate_and_plugin_strings_outside_memory_too() -> None:
    for expression in (
        "escapeHtml(p.title)",
        "escapeHtml(field.label)",
        "escapeHtml(report.candidate_id)",
        "escapeHtml(promotion.reason)",
        "escapeHtml(selectedName)",
        "escapeHtml(r.promotion?.reason||'')",
        "escapeHtml(page.result.card.title)",
        "escapeHtml(page.result.card.summary)",
        "escapeHtml(page.result.usage)",
    ):
        assert expression in HTML


def test_waiting_for_model_service_clears_only_after_all_agents_resume():
    from pde_agent_verifier_solver.engine.runtime import RunEvent
    from pde_agent_verifier_solver.panel.controller import _agent_recovery_activity

    def event(index, kind, agent):
        return RunEvent(index, kind, "Model service progress", 1, {"agent_id": agent})

    events = [event(1, "agent_recovering", "a"), event(2, "agent_recovering", "b")]
    assert _agent_recovery_activity(events) == "Waiting for model service; saved Agent work will resume"
    events.append(event(3, "agent_resumed", "a"))
    assert _agent_recovery_activity(events) is not None
    events.append(event(4, "agent_finished", "b"))
    assert _agent_recovery_activity(events) is None
    events.extend([event(5, "agent_recovering", "a"), event(6, "agent_failed", "a")])
    assert _agent_recovery_activity(events) is None
    assert "$('status').textContent=page.activity || page.status" in HTML


def test_native_recovery_progress_uses_only_safe_status_fields(tmp_path, monkeypatch):
    import json
    from pde_agent_verifier_solver.engine import agent_execution as execution

    clock = iter([0.0, 2.0, 4.0, 6.0, 8.0])
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(clock))
    seen = []
    observe = execution._native_recovery_observer(tmp_path, lambda state, fields: seen.append((state, fields)))
    path = tmp_path / "codex-recovery.json"
    path.write_text("not yet complete")
    observe()
    record = {"schema": "canonical_codex_same_session_recovery_v1", "status": "retry",
              "current_attempt": 1, "attempts": [{"failure": "private provider text"}]}
    path.write_text(json.dumps(record))
    observe()
    observe()
    record.update(status="running", current_attempt=2)
    path.write_text(json.dumps(record))
    observe()
    path.unlink()
    path.symlink_to(tmp_path / "missing")
    observe()
    assert seen == [("recovering", {"attempt": 1}), ("resumed", {"attempt": 2})]


def test_recovery_display_failure_does_not_interrupt_agent_execution(tmp_path, monkeypatch):
    import json
    from pde_agent_verifier_solver.engine import agent_execution as execution

    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(ticks))
    calls = []

    def unavailable_display(state, fields):
        calls.append((state, fields))
        raise RuntimeError("display subscriber is unavailable")

    (tmp_path / "codex-recovery.json").write_text(json.dumps({
        "schema": "canonical_codex_same_session_recovery_v1",
        "status": "retry", "current_attempt": 1,
    }))
    observe = execution._native_recovery_observer(tmp_path, unavailable_display)
    observe()
    observe()
    assert calls == [("recovering", {"attempt": 1})] * 2
