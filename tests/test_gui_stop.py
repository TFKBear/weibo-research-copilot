"""停止采集后，完成回调不得再次启动自动分析。"""
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from wrc.gui.app import App


@pytest.mark.parametrize("stopped", [False, True])
def test_collection_callback_respects_stop_before_auto_analysis(monkeypatch, stopped):
    monkeypatch.setattr("wrc.gui.app.messagebox.askyesno", lambda *_a, **_k: False)
    stop = Event()
    if stopped:
        stop.set()
    analyses = []
    true_var = SimpleNamespace(get=lambda: True)
    app = SimpleNamespace(
        _checked_posts=lambda: [object()], consent_var=true_var,
        _window_from_ui=lambda: None,
        _clean_options=lambda: SimpleNamespace(use_ai=False, tier="moderate"),
        _collect_options=lambda: object(), _log=lambda *_a: None,
        stop_event=stop, auto_analyze_var=true_var,
        do_sentiment_var=true_var, do_cluster_var=true_var, do_stance_var=true_var,
        _on_analyze=lambda: analyses.append(True),
        _start_job=lambda job, done: done({
            "run_id": "test", "comments": 2, "kept": 2, "rate": 1,
            "out_dir": Path("unused"), "appendix": None,
        }),
    )
    App._on_start(app)
    assert bool(analyses) is not stopped
