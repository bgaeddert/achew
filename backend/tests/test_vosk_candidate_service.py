"""Unit tests for bounded, ordered Vosk candidate scanning."""

import asyncio
import threading
import time

from app.services import vosk_candidate_service
from app.services.vosk_candidate_service import VoskCandidateService, VoskWord


def test_default_terms_use_supported_librivox_suffix():
    assert "vox" in vosk_candidate_service.TERMS
    assert "libri" not in vosk_candidate_service.TERMS
    assert "librivox" not in vosk_candidate_service.TERMS


def test_model_downloads_by_name_when_no_path_is_configured(monkeypatch):
    model = object()
    calls = []

    monkeypatch.setattr(vosk_candidate_service, "_model", None)
    monkeypatch.setattr(vosk_candidate_service, "MODEL_PATH", None)
    monkeypatch.setattr(
        vosk_candidate_service,
        "_get_vosk_api",
        lambda: vosk_candidate_service._VoskAPI(
            lambda *args, **kwargs: calls.append((args, kwargs)) or model,
            object(),
            lambda _level: None,
        ),
    )

    assert vosk_candidate_service._get_model() is model
    assert calls == [((), {"model_name": vosk_candidate_service.MODEL_NAME})]


def test_configured_model_path_takes_precedence(monkeypatch, tmp_path):
    model = object()
    calls = []

    monkeypatch.setattr(vosk_candidate_service, "_model", None)
    monkeypatch.setattr(vosk_candidate_service, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(
        vosk_candidate_service,
        "_get_vosk_api",
        lambda: vosk_candidate_service._VoskAPI(
            lambda *args, **kwargs: calls.append((args, kwargs)) or model,
            object(),
            lambda _level: None,
        ),
    )

    assert vosk_candidate_service._get_model() is model
    assert calls == [((str(tmp_path),), {})]


def test_vosk_is_unavailable_on_native_macos_without_checking_for_package(monkeypatch):
    monkeypatch.setattr(vosk_candidate_service.sys, "platform", "darwin")
    monkeypatch.setattr(
        vosk_candidate_service.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(AssertionError("native macOS must be rejected first")),
    )

    assert not vosk_candidate_service.is_vosk_available()


def test_vosk_is_available_in_linux_container(monkeypatch):
    monkeypatch.setattr(vosk_candidate_service.sys, "platform", "linux")
    monkeypatch.setattr(
        vosk_candidate_service.importlib.util, "find_spec", lambda name: object() if name == "vosk" else None
    )

    assert vosk_candidate_service.is_vosk_available()


def test_lazy_vosk_import_rejects_native_macos(monkeypatch):
    monkeypatch.setattr(vosk_candidate_service, "_vosk_api", None)
    monkeypatch.setattr(vosk_candidate_service.sys, "platform", "darwin")

    try:
        vosk_candidate_service._get_vosk_api()
    except RuntimeError as error:
        assert str(error) == "Intelligent chapter detection is not available on native macOS"
    else:
        raise AssertionError("native macOS should not load Vosk")


def test_collect_loads_model_before_scanning(monkeypatch):
    events = []
    service = VoskCandidateService(lambda *args, **kwargs: None)

    monkeypatch.setattr(vosk_candidate_service, "_get_model", lambda: events.append("model"))
    monkeypatch.setattr(service, "_scan", lambda *args: events.append("scan") or [])

    assert asyncio.run(service.collect("book.m4b", 100.0, [10.0])) == []
    assert events == ["model", "scan"]


def test_parallel_scan_bounds_workers_and_preserves_time_order(monkeypatch):
    """Out-of-order workers must not alter the evidence supplied to triage."""
    active = 0
    peak_active = 0
    active_lock = threading.Lock()
    progress = []

    def record_progress(step, percent, message="", details=None):
        progress.append((percent, message, details or {}))

    service = VoskCandidateService(record_progress, terms=["chapter", "part", "section"])
    monkeypatch.setattr("app.services.vosk_candidate_service.get_worker_count", lambda: 2)

    def fake_scan_window(audio_file, start, end, cancelled):
        nonlocal active, peak_active
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
        # Finish later windows first to exercise completion-order handling.
        time.sleep((40.0 - start) / 1_000)
        with active_lock:
            active -= 1
        return [VoskWord(word="chapter", start=start + 0.1, end=start + 0.2, confidence=0.9)]

    monkeypatch.setattr(service, "_scan_window", fake_scan_window)

    evidence = service._scan("book.m4b", 100.0, [10.0, 20.0, 30.0], threading.Event())

    assert peak_active == 2
    assert [row.timestamp for row in evidence] == [10.0, 20.0, 30.0]
    assert [[word.start for word in row.words] for row in evidence] == [[9.6], [19.6], [29.6]]
    completed = [details["regions_completed"] for _, _, details in progress]
    assert completed == [1, 2, 3]
    assert all(details["worker_count"] == 2 for _, _, details in progress)


def test_cancelled_scan_does_not_start_a_vosk_decoder(monkeypatch):
    """A queued worker observes cancellation before loading Vosk or ffmpeg."""
    service = VoskCandidateService(lambda *args, **kwargs: None)
    cancelled = threading.Event()
    cancelled.set()

    monkeypatch.setattr(
        "app.services.vosk_candidate_service._get_model",
        lambda: (_ for _ in ()).throw(AssertionError("model must not be loaded after cancellation")),
    )

    assert service._scan_window("book.m4b", 10.0, 13.0, cancelled) == []
