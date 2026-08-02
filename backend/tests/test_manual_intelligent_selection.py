from types import SimpleNamespace

import pytest

from app.api.routes import pipeline as pipeline_routes
from app.models.enums import Step
from app.models.references import ReferenceValidationChapter, ReferenceValidationResult
from app.services.processing_pipeline import INTELLIGENT_DETECTION_RESULT_ID, ProcessingPipeline


@pytest.mark.asyncio
async def test_manual_intelligent_selection_updates_result_and_queues_asr(monkeypatch):
    pipeline = object.__new__(ProcessingPipeline)
    pipeline.step = Step.REFERENCE_VALIDATION_RESULTS
    pipeline.progress_callback = lambda *args: None
    pipeline._filter_cues_by_duration = lambda cues: cues
    result = ReferenceValidationResult(
        id=INTELLIGENT_DETECTION_RESULT_ID,
        name="Intelligent Chapter Detection",
        short_name="Intelligent Detection",
        description="test",
        type="intelligent_detection",
        chapters=[
            ReferenceValidationChapter(timestamp=10.0, title="Candidate 1"),
            ReferenceValidationChapter(timestamp=20.0, title="Candidate 2"),
            ReferenceValidationChapter(timestamp=30.0, title="Candidate 3"),
        ],
        duration=100.0,
    )
    pipeline._reference_validation_results = [result]
    pipeline._intelligent_validation_result = result

    cues = await pipeline.apply_intelligent_detection_selection([0, 2])

    assert cues == [0.0, 10.0, 30.0]
    assert pipeline.cues == cues
    assert pipeline.step == Step.CONFIGURE_ASR
    assert [chapter.valid for chapter in result.chapters] == [True, False, True]


@pytest.mark.asyncio
async def test_manual_intelligent_selection_rejects_duplicate_ids():
    pipeline = object.__new__(ProcessingPipeline)
    pipeline.step = Step.REFERENCE_VALIDATION_RESULTS

    with pytest.raises(ValueError, match="must be unique"):
        await pipeline.apply_intelligent_detection_selection([1, 1])


@pytest.mark.asyncio
async def test_manual_selection_endpoint_returns_selected_cues(monkeypatch):
    seen: dict[str, list[int]] = {}

    async def apply_selection(candidate_ids: list[int]) -> list[float]:
        seen["candidate_ids"] = candidate_ids
        return [0.0, 42.5]

    pipeline = SimpleNamespace(
        step=Step.REFERENCE_VALIDATION_RESULTS,
        apply_intelligent_detection_selection=apply_selection,
    )
    monkeypatch.setattr(pipeline_routes, "is_vosk_available", lambda: True)
    monkeypatch.setattr(
        pipeline_routes,
        "get_app_state",
        lambda: SimpleNamespace(pipeline=pipeline),
    )

    result = await pipeline_routes.apply_manual_intelligent_detection_selection(
        pipeline_routes.IntelligentDetectionManualSelectionRequest(
            selected_candidate_ids=[2, 5]
        )
    )

    assert seen == {"candidate_ids": [2, 5]}
    assert result["selected_candidate_ids"] == [2, 5]
    assert result["selected_timestamps"] == [0.0, 42.5]
    assert result["next_step"] == Step.CONFIGURE_ASR.value
