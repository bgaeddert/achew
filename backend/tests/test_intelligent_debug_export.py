from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes import pipeline as pipeline_routes
from app.models.enums import Step
from app.services import processing_pipeline
from app.services.processing_pipeline import ProcessingPipeline


async def test_intelligent_debug_export_is_hidden_without_debug(monkeypatch):
    monkeypatch.setattr(pipeline_routes, "get_settings", lambda: SimpleNamespace(DEBUG=False))

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_routes.export_intelligent_chapter_detection_llm_debug()

    assert exc_info.value.status_code == 404


async def test_intelligent_debug_export_requires_an_in_memory_capture(monkeypatch):
    monkeypatch.setattr(pipeline_routes, "get_settings", lambda: SimpleNamespace(DEBUG=True))
    monkeypatch.setattr(
        pipeline_routes,
        "get_app_state",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(_intelligent_llm_debug=None)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_routes.export_intelligent_chapter_detection_llm_debug()

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "No intelligent-detection LLM capture is available"


async def test_intelligent_debug_export_returns_latest_capture(monkeypatch):
    capture = {
        "item_id": "book-1",
        "provider_id": "openai",
        "request": {"candidate_evidence": [["chapter", "one"]]},
        "response": {"keep": [True]},
    }
    monkeypatch.setattr(pipeline_routes, "get_settings", lambda: SimpleNamespace(DEBUG=True))
    monkeypatch.setattr(
        pipeline_routes,
        "get_app_state",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(_intelligent_llm_debug=capture)),
    )

    result = await pipeline_routes.export_intelligent_chapter_detection_llm_debug()

    assert result["capture"] == capture
    assert result["filename"].startswith("intelligent_detection_llm_debug_")
    assert result["filename"].endswith(".json")


async def test_intelligent_detection_entry_point_is_unavailable_without_vosk(monkeypatch):
    monkeypatch.setattr(pipeline_routes, "is_vosk_available", lambda: False)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_routes.open_intelligent_chapter_detection()

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Intelligent chapter detection is not available on this platform"


async def test_detected_cues_report_intelligent_detection_availability(monkeypatch):
    pipeline = SimpleNamespace(
        detected_cues=[SimpleNamespace(timestamp=10.0, gap=2.5)],
        book_duration=100.0,
        chapter_refs=[],
    )
    monkeypatch.setattr(
        pipeline_routes,
        "get_app_state",
        lambda: SimpleNamespace(step=Step.INITIAL_CHAPTER_SELECTION, pipeline=pipeline),
    )
    monkeypatch.setattr(pipeline_routes, "is_vosk_available", lambda: False)

    result = await pipeline_routes.get_detected_cues()

    assert result["intelligent_chapter_detection_available"] is False


def test_intelligent_debug_capture_is_kept_only_in_debug_mode(monkeypatch):
    pipeline = object.__new__(ProcessingPipeline)
    pipeline.item_id = "book-1"
    pipeline._intelligent_llm_debug = {"old": "capture"}
    monkeypatch.setattr(processing_pipeline, "get_settings", lambda: SimpleNamespace(DEBUG=False))

    pipeline._capture_intelligent_llm_debug({"provider_id": "openai"}, {"response": {"keep": [True]}})

    assert pipeline._intelligent_llm_debug is None


def test_intelligent_debug_capture_combines_metadata_and_provider_exchange(monkeypatch):
    pipeline = object.__new__(ProcessingPipeline)
    pipeline.item_id = "book-1"
    pipeline._intelligent_llm_debug = None
    monkeypatch.setattr(processing_pipeline, "get_settings", lambda: SimpleNamespace(DEBUG=True))

    pipeline._capture_intelligent_llm_debug(
        {"provider_id": "openai", "model_id": "gpt-test"},
        {"request": {"prompt": "test"}, "response": {"keep": [True]}},
    )

    assert pipeline._intelligent_llm_debug == {
        "item_id": "book-1",
        "provider_id": "openai",
        "model_id": "gpt-test",
        "request": {"prompt": "test"},
        "response": {"keep": [True]},
    }
