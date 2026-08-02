import logging
import traceback
from datetime import datetime
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.models.references import ChapterReference, ReferenceValidationResult, TitleReference
from app.services.processing_pipeline import PipelineProgress, ProcessingPipeline
from app.services.vosk_candidate_service import TERMS, is_vosk_available

from ...app import get_app_state
from ...core.config import get_app_config, get_settings, is_abs_configured, save_llm_config
from ...models.abs import AudioInfo, Book
from ...models.enums import DetectionMode, RestartStep, Step

logger = logging.getLogger(__name__)

router = APIRouter()


class CreatePipelineRequest(BaseModel):
    item_id: str


class StartWorkflowRequest(BaseModel):
    workflow: str
    ref_id: Optional[str] = None
    detection_mode: DetectionMode = DetectionMode.STANDARD
    thorough: Optional[bool] = False


class DramatizedFixtureRequest(BaseModel):
    ground_truth: Literal["standard", "dramatized"]


class RestartPipelineRequest(BaseModel):
    restart_step: RestartStep


class CancelStepRequest(BaseModel):
    expected_step: Optional[Step] = None


class ASROptionsRequest(BaseModel):
    trim: bool
    use_bias_words: bool = False
    bias_words: str = ""
    segment_length: float = 8.0


class PreassignedTitle(BaseModel):
    cue_index: int
    title: str


class ConfigureASRRequest(BaseModel):
    action: str  # "transcribe" or "skip"
    preassigned_titles: List[PreassignedTitle] = []


class IntelligentChapterDetectionRequest(BaseModel):
    action: Literal["run", "skip", "update_terms"]
    provider_id: str = ""
    model_id: str = ""
    reference_id: str = ""
    vosk_terms: List[str] = []
    minimum_pause_seconds: float = 2.0
    post_pause_seconds: float = 1.0
    vosk_clip_length: float = 3.0
    llm_triage: bool = True
    view_results: bool = False
    validate_references: bool = False


class IntelligentChapterDetectionOptionsResponse(BaseModel):
    base_terms: List[str]
    chapter_refs: List[ChapterReference]
    vosk_terms: List[str]
    minimum_pause_seconds: float
    post_pause_seconds: float
    vosk_clip_length: float
    quick_validate: bool
    llm_processing: bool
    provider_id: str
    model_id: str


class ReferenceValidationResultsResponse(BaseModel):
    references: List[ReferenceValidationResult]


class ReferenceValidationTranscriptionRequest(BaseModel):
    reference_id: str


class IntelligentDetectionManualSelectionRequest(BaseModel):
    """Candidate indexes retained after manual review of Vosk results."""

    selected_candidate_ids: List[int] = Field(min_length=1)


def _require_intelligent_chapter_detection() -> None:
    if not is_vosk_available():
        raise HTTPException(status_code=404, detail="Intelligent chapter detection is not available on this platform")


def _save_llm_selection(pipeline: ProcessingPipeline, provider_id: str, model_id: str) -> None:
    """Persist a pipeline's LLM picker selection using the shared AI preferences."""
    config = get_app_config()
    config.llm.last_used_provider = provider_id
    config.llm.last_used_model = model_id
    if not save_llm_config(config.llm):
        raise HTTPException(status_code=500, detail="Failed to save LLM provider and model preference")
    pipeline.ai_options.provider_id = provider_id
    pipeline.ai_options.model_id = model_id


class PipelineStateResponse(BaseModel):
    item_id: str
    step: str
    progress: PipelineProgress
    selection_stats: Dict[str, int]
    can_undo: bool
    can_redo: bool
    book: Optional[Book] = None
    chapter_refs: List[ChapterReference] = []
    title_refs: List[TitleReference] = []
    restart_options: List[str] = []
    audio_unsupported_codec: bool = False
    audio_info: Optional[AudioInfo] = None
    intelligent_chapter_detection_available: bool = False


@router.post("/pipeline", response_model=dict)
async def create_pipeline(request: CreatePipelineRequest, background_tasks: BackgroundTasks):
    """Create a new processing pipeline"""
    # Check if API is configured
    if not is_abs_configured():
        raise HTTPException(
            status_code=400,
            detail="ABS configuration required. Please configure ABS API key first.",
        )

    try:
        app_state = get_app_state()
        pipeline = app_state.create_pipeline(request.item_id)

        # Start processing in background
        async def start_processing():
            try:
                result = await pipeline.fetch_item(request.item_id)
                logger.info(f"Fetched item: {result}")

                await app_state._broadcast_book_update()

                await app_state.broadcast_step_change(
                    Step.SELECT_WORKFLOW,
                    extras={
                        "chapter_refs": pipeline.chapter_refs,
                        "title_refs": pipeline.title_refs,
                        "audio_unsupported_codec": pipeline.audio_unsupported_codec,
                        "audio_info": pipeline.audio_info,
                        "intelligent_chapter_detection_available": is_vosk_available(),
                    },
                )

            except Exception as e:
                logger.error(f"Fetching item failed: {e}", exc_info=True)

        background_tasks.add_task(start_processing)

        return {
            "message": "Pipeline created and processing started",
        }

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/state", response_model=PipelineStateResponse)
async def get_pipeline_state():
    """Get pipeline state details"""
    try:
        app_state = get_app_state()

        if not app_state.pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        pipeline = app_state.pipeline

        # Get stats from pipeline
        stats = pipeline.get_selection_stats()

        progress = PipelineProgress(
            step=pipeline.step,
            percent=0.0,  # Would need to get from pipeline
            message="",  # Would need to get from pipeline
            details={},
        )

        return PipelineStateResponse(
            item_id=pipeline.item_id,
            step=app_state.step.value,
            progress=progress,
            selection_stats=stats,
            can_undo=pipeline.can_undo(),
            can_redo=pipeline.can_redo(),
            book=pipeline.book if pipeline.book else None,
            chapter_refs=pipeline.chapter_refs,
            title_refs=pipeline.title_refs,
            restart_options=pipeline.get_restart_options(),
            audio_unsupported_codec=pipeline.audio_unsupported_codec,
            audio_info=pipeline.audio_info,
            intelligent_chapter_detection_available=is_vosk_available(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get app state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pipeline")
async def delete_pipeline():
    """Delete the current pipeline and cleanup resources"""
    try:
        app_state = get_app_state()
        success = await app_state.delete_pipeline()

        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        return {"message": "Pipeline deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/submit")
async def submit_chapters():
    """Submit chapters to Audiobookshelf"""
    try:
        app_state = get_app_state()
        pipeline = app_state.pipeline

        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        # Allow submission from chapter_editing or reviewing step
        if pipeline.step not in [Step.CHAPTER_EDITING, Step.REVIEWING]:
            raise HTTPException(
                status_code=400,
                detail="Pipeline must be in chapter_editing or reviewing step to submit",
            )

        success = await pipeline.submit_chapters(pipeline.chapters)

        if success:
            pipeline.step = Step.COMPLETED
            await app_state.broadcast_step_change(Step.COMPLETED)
            return {"message": "Chapters submitted successfully"}
        else:
            pipeline.step = Step.REVIEWING
            await app_state.broadcast_step_change(Step.REVIEWING)
            raise HTTPException(status_code=500, detail="Failed to submit chapters")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit chapters for pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/goto-review")
async def goto_review():
    """Transition from chapter editing to reviewing step"""
    try:
        app_state = get_app_state()
        pipeline = app_state.pipeline

        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        if pipeline.step != Step.CHAPTER_EDITING:
            raise HTTPException(
                status_code=400,
                detail="Pipeline must be in chapter_editing step to go to review",
            )

        pipeline.step = Step.REVIEWING
        await app_state.broadcast_step_change(Step.REVIEWING)

        return {"message": "Transitioned to reviewing", "step": Step.REVIEWING.value}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to transition to reviewing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/start-workflow")
async def start_workflow(request: StartWorkflowRequest, background_tasks: BackgroundTasks):
    """Set workflow"""
    try:
        app_state = get_app_state()

        if not app_state.pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        pipeline = app_state.pipeline
        quick_edit_from_validation = (
            app_state.step == Step.REFERENCE_VALIDATION_RESULTS and request.workflow == "quick_edit"
        )
        if app_state.step != Step.SELECT_WORKFLOW and not quick_edit_from_validation:
            raise HTTPException(
                status_code=400,
                detail="Pipeline must be in select_workflow step to select option",
            )

        if quick_edit_from_validation:
            validated_result = next(
                (result for result in pipeline._reference_validation_results if result.id == request.ref_id),
                None,
            )
            if validated_result is None:
                raise HTTPException(
                    status_code=400,
                    detail="Choose one of the validated results to quick edit",
                )

        async def run_workflow():
            try:
                await pipeline.start_workflow(
                    request.workflow, request.ref_id, request.detection_mode, request.thorough
                )
            except Exception as e:
                logger.error(f"Failed to start workflow: {e}")

        background_tasks.add_task(run_workflow)

        return {
            "message": f"Selected workflow '{request.workflow}'",
            "workflow": request.workflow,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select workflow for pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/dramatized-fixture")
async def export_dramatized_fixture(request: DramatizedFixtureRequest):
    """DEBUG-only: run the dramatized auto-detection probe on the current book and export
    it as a regression test fixture tagged with the user-supplied ground-truth label.

    Runs the probe silently (does not change the pipeline step / UI). Disabled (404)
    unless DEBUG is set.
    """
    if not get_settings().DEBUG:
        raise HTTPException(status_code=404, detail="Not found")

    try:
        app_state = get_app_state()
        pipeline = app_state.pipeline

        if not pipeline:
            raise HTTPException(status_code=404, detail="No active pipeline")

        if app_state.step != Step.SELECT_WORKFLOW:
            raise HTTPException(
                status_code=400,
                detail="Pipeline must be in select_workflow step to export a dramatized fixture",
            )

        # Always run fresh so captures are reproducible (bypasses the session cache).
        analysis = await pipeline._run_dramatized_probe(broadcast=False)
        if analysis is None or pipeline._dramatized_probe is None:
            raise HTTPException(status_code=500, detail="Dramatized probe did not complete")

        ground_truth_dramatized = request.ground_truth == "dramatized"

        fixture = {
            **pipeline._dramatized_probe,
            "ground_truth_dramatized": ground_truth_dramatized,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ``computed`` is returned for the debug UI's agree/disagree readout only; the
        # frontend writes just ``fixture`` to the downloaded file.
        return {
            "fixture": fixture,
            "computed": {
                "is_dramatized": analysis.is_dramatized,
                "standard_cue_count": analysis.standard_cue_count,
                "vad_cue_count": analysis.vad_cue_count,
                "unmatched_notable_cues": [list(cue) for cue in analysis.unmatched_notable_cues],
            },
            "filename": f"dramatized_fixture_{timestamp}.json",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export dramatized fixture: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/pipeline/intelligent-chapter-detection/options", response_model=IntelligentChapterDetectionOptionsResponse
)
async def get_intelligent_chapter_detection_options():
    """Return the editable default Vosk grammar and available chapter references."""
    _require_intelligent_chapter_detection()
    app_state = get_app_state()
    pipeline = app_state.pipeline
    if pipeline and app_state.step not in [Step.INTELLIGENT_CHAPTER_DETECTION, Step.INTELLIGENT_DETECTION_SETUP]:
        raise HTTPException(status_code=400, detail="Pipeline is not ready for intelligent chapter detection")
    config = get_app_config()
    settings = config.intelligent_detection
    return IntelligentChapterDetectionOptionsResponse(
        base_terms=TERMS,
        chapter_refs=pipeline.chapter_refs if pipeline else [],
        vosk_terms=settings.vosk_terms,
        minimum_pause_seconds=settings.minimum_pause_seconds,
        post_pause_seconds=settings.post_pause_seconds,
        vosk_clip_length=settings.vosk_clip_length,
        quick_validate=settings.quick_validate,
        llm_processing=settings.llm_processing,
        provider_id=config.llm.last_used_provider,
        model_id=config.llm.last_used_model,
    )


@router.get("/pipeline/intelligent-chapter-detection/llm-debug")
async def export_intelligent_chapter_detection_llm_debug():
    """DEBUG-only: export the latest in-memory intelligent-detection LLM exchange."""
    if not get_settings().DEBUG:
        raise HTTPException(status_code=404, detail="Not found")

    app_state = get_app_state()
    pipeline = app_state.pipeline
    if not pipeline:
        raise HTTPException(status_code=404, detail="No active pipeline")
    if not pipeline._intelligent_llm_debug:
        raise HTTPException(status_code=404, detail="No intelligent-detection LLM capture is available")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "capture": pipeline._intelligent_llm_debug,
        "filename": f"intelligent_detection_llm_debug_{timestamp}.json",
    }


@router.post("/pipeline/intelligent-chapter-detection")
async def intelligent_chapter_detection(
    request: IntelligentChapterDetectionRequest,
    background_tasks: BackgroundTasks,
):
    """Run, bypass, or validate references with the optional Vosk/LLM workflow."""
    try:
        _require_intelligent_chapter_detection()
        app_state = get_app_state()
        pipeline = app_state.pipeline
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        if pipeline.step != Step.INTELLIGENT_CHAPTER_DETECTION:
            raise HTTPException(status_code=400, detail="Pipeline is not ready for intelligent chapter detection")

        if request.action == "skip":
            await pipeline.skip_intelligent_chapter_detection()
            return {"message": "Skipped intelligent chapter detection"}

        # Keep the picker consistent with AI cleanup: the last provider/model
        # selected for intelligent detection becomes the default next time the
        # page is opened, including after an LLM retry.
        _save_llm_selection(pipeline, request.provider_id, request.model_id)

        if request.action == "update_terms":
            terms = await pipeline.suggest_intelligent_vosk_terms(
                request.provider_id,
                request.model_id,
                request.reference_id,
                request.vosk_terms,
            )
            return {"terms": terms}

        async def run_detection():
            try:
                llm_processing = request.llm_triage
                validate_references = request.validate_references
                if request.view_results or validate_references:
                    await pipeline.run_intelligent_detection_for_results(
                        request.provider_id,
                        request.model_id,
                        request.vosk_terms,
                        request.minimum_pause_seconds,
                        request.post_pause_seconds,
                        request.vosk_clip_length,
                        use_llm=llm_processing,
                        validate_references=validate_references,
                    )
                else:
                    await pipeline.run_intelligent_chapter_detection(
                        request.provider_id,
                        request.model_id,
                        request.vosk_terms,
                        request.minimum_pause_seconds,
                        request.post_pause_seconds,
                        request.vosk_clip_length,
                        use_llm=llm_processing,
                    )
            except Exception as e:
                logger.error("Intelligent chapter detection failed: %s", e, exc_info=True)
                pipeline.step = Step.INTELLIGENT_CHAPTER_DETECTION
                await app_state.broadcast_step_change(
                    Step.INTELLIGENT_CHAPTER_DETECTION,
                    error_message=f"Intelligent chapter detection failed: {e}",
                )

        background_tasks.add_task(run_detection)
        return {"message": "Intelligent chapter detection started"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start intelligent chapter detection: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/pipeline/reference-validation/results",
    response_model=ReferenceValidationResultsResponse,
)
async def get_reference_validation_results():
    """Return read-only Vosk and LLM checks for each timed chapter reference."""
    _require_intelligent_chapter_detection()
    pipeline = get_app_state().pipeline
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    if pipeline.step not in {Step.REFERENCE_VALIDATION_RESULTS, Step.SELECT_WORKFLOW}:
        raise HTTPException(status_code=400, detail="Chapter reference validation results are not ready")
    return ReferenceValidationResultsResponse(references=pipeline._reference_validation_results)


@router.post("/pipeline/intelligent-chapter-detection/manual-selection")
async def apply_manual_intelligent_detection_selection(
    request: IntelligentDetectionManualSelectionRequest,
):
    """Apply manually reviewed intelligent-detection candidates and continue to ASR."""
    _require_intelligent_chapter_detection()
    pipeline = get_app_state().pipeline
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    if pipeline.step != Step.REFERENCE_VALIDATION_RESULTS:
        raise HTTPException(
            status_code=400,
            detail="Intelligent detection results are not ready for manual selection",
        )
    try:
        selected_timestamps = await pipeline.apply_intelligent_detection_selection(
            request.selected_candidate_ids
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "message": "Manual intelligent-detection selection applied; ready for transcription",
        "selected_candidate_ids": request.selected_candidate_ids,
        "selected_timestamps": selected_timestamps,
        "next_step": Step.CONFIGURE_ASR.value,
    }


@router.get(
    "/pipeline/reference-validation/vosk-results",
    response_model=ReferenceValidationResultsResponse,
)
async def get_reference_vosk_results():
    """Return the automatic Vosk-only scores shown on workflow reference cards."""
    pipeline = get_app_state().pipeline
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    if pipeline.step != Step.SELECT_WORKFLOW:
        raise HTTPException(status_code=400, detail="Chapter reference scores are not ready")
    return ReferenceValidationResultsResponse(references=pipeline._reference_vosk_results)


@router.post("/pipeline/reference-validation/results/transcribe")
async def transcribe_validated_reference(request: ReferenceValidationTranscriptionRequest):
    """Use every timestamp from one validated reference as the transcription cue list."""
    _require_intelligent_chapter_detection()
    pipeline = get_app_state().pipeline
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    if pipeline.step != Step.REFERENCE_VALIDATION_RESULTS:
        raise HTTPException(status_code=400, detail="Chapter reference validation results are not ready")
    try:
        await pipeline.prepare_reference_for_transcription(request.reference_id)
        return {
            "message": "Reference chapters are ready for transcription",
            "reference_id": request.reference_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/pipeline/open-intelligent-chapter-detection")
async def open_intelligent_chapter_detection():
    """Open the optional Vosk/LLM timeline cleanup page from initial selection."""
    try:
        _require_intelligent_chapter_detection()
        app_state = get_app_state()
        pipeline = app_state.pipeline
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        if pipeline.step != Step.INITIAL_CHAPTER_SELECTION:
            raise HTTPException(
                status_code=400, detail="Timeline cleanup is only available from initial chapter selection"
            )
        await pipeline._transition_to_intelligent_chapter_detection()
        return {"message": "Opened intelligent chapter detection"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to open intelligent chapter detection: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/detected-cues")
async def get_detected_cues():
    """Get all detected cues for initial chapter selection"""
    try:
        app_state = get_app_state()

        if not app_state.pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        if app_state.step != Step.INITIAL_CHAPTER_SELECTION:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline not in initial chapter selection step. Current step: {app_state.step.value}",
            )

        if not app_state.pipeline.detected_cues:
            raise HTTPException(status_code=400, detail="No detected cues available")

        # Sort by timestamp
        detected_cues = app_state.pipeline.detected_cues.copy()
        detected_cues.sort(key=lambda x: x.timestamp)

        return {
            "detected_cues": detected_cues,
            "book_duration": app_state.pipeline.book_duration,
            "chapter_refs": app_state.pipeline.chapter_refs,
            "intelligent_chapter_detection_available": is_vosk_available(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detected cues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/select-initial-chapters")
async def select_initial_chapters(request: dict, background_tasks: BackgroundTasks):
    """Select initial chapters by providing a list of cue timestamps"""
    try:
        app_state = get_app_state()

        if not app_state.pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        if app_state.step != Step.INITIAL_CHAPTER_SELECTION:
            raise HTTPException(status_code=400, detail="Pipeline not in initial chapter selection step")

        timestamps = request.get("timestamps")
        if timestamps is None or not isinstance(timestamps, list):
            raise HTTPException(status_code=400, detail="timestamps must be a list of floats")
        if len(timestamps) == 0:
            raise HTTPException(status_code=400, detail="At least one timestamp is required")

        include_unaligned = request.get("include_unaligned", [])
        if not isinstance(include_unaligned, list):
            raise HTTPException(status_code=400, detail="include_unaligned must be a list")

        # Validate each option against available chapter references
        available_ref_ids = [ref.id for ref in app_state.pipeline.chapter_refs]
        for option in include_unaligned:
            if option not in available_ref_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid include_unaligned option: {option}. Available options: {available_ref_ids}",
                )

        pipeline = app_state.pipeline

        async def do_select_initial_chapters():
            try:
                await pipeline.select_initial_chapters(timestamps, include_unaligned)
            except Exception as e:
                logger.error(f"Failed to select initial chapters: {e}")

        background_tasks.add_task(do_select_initial_chapters)

        return {
            "message": "Initial chapters selected, extracting segments…",
            "include_unaligned": include_unaligned,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select initial chapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/configure-asr")
async def configure_asr(request: ConfigureASRRequest, background_tasks: BackgroundTasks):
    """Configure ASR settings and proceed with transcription or skip"""
    try:
        app_state = get_app_state()

        if not app_state.pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        if app_state.step != Step.CONFIGURE_ASR:
            raise HTTPException(
                status_code=400,
                detail="Pipeline must be in configure_asr step to configure ASR",
            )

        # Validate action
        if request.action not in ["transcribe", "skip"]:
            raise HTTPException(
                status_code=400,
                detail="Action must be 'transcribe' or 'skip'",
            )

        pipeline = app_state.pipeline

        # Validate preassigned titles and build the cue_index -> title map
        cue_count = len(pipeline.cues)
        preassigned: Dict[int, str] = {}
        for item in request.preassigned_titles:
            if item.cue_index < 0 or item.cue_index >= cue_count:
                raise HTTPException(
                    status_code=400,
                    detail=f"preassigned_titles cue_index {item.cue_index} out of range [0, {cue_count})",
                )
            if item.cue_index in preassigned:
                raise HTTPException(
                    status_code=400,
                    detail=f"preassigned_titles contains duplicate cue_index {item.cue_index}",
                )
            preassigned[item.cue_index] = item.title

        async def process_asr_action():
            try:
                if request.action == "transcribe":
                    await pipeline.proceed_with_transcription(preassigned_titles=preassigned)
                elif request.action == "skip":
                    await pipeline.skip_transcription(preassigned_titles=preassigned)
            except Exception as e:
                logger.error(f"Failed to process ASR action: {e}")

        background_tasks.add_task(process_asr_action)

        return {
            "message": f"ASR action '{request.action}' initiated",
            "action": request.action,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure ASR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/selected-cues")
async def get_selected_cues():
    """Get the cues that are queued for transcription"""
    try:
        app_state = get_app_state()

        if not app_state.pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        if app_state.step != Step.CONFIGURE_ASR:
            raise HTTPException(
                status_code=400,
                detail="Pipeline must be in configure_asr step to get selected cues",
            )

        return {
            "cues": list(app_state.pipeline.cues),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get selected cues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/restart")
async def restart_pipeline(request: RestartPipelineRequest):
    """Restart pipeline with selective cleanup"""
    try:
        app_state = get_app_state()
        if not app_state.pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        await app_state.pipeline.restart_at_step(request.restart_step)

        return {
            "message": f"Pipeline restarted at step '{request.restart_step.value}'",
            "restart_step": request.restart_step.value,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/cancel")
async def cancel_step(request: Optional[CancelStepRequest] = None):
    """Cancel the current processing pipeline step and return to the appropriate previous step"""
    try:
        app_state = get_app_state()
        pipeline = app_state.pipeline

        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        step = pipeline.step
        expected_step = request.expected_step if request else None
        intelligent_steps = [Step.VOSK_ANALYSIS, Step.LLM_CANDIDATE_TRIAGE, Step.REFERENCE_VALIDATION]

        if step in [Step.VALIDATING, Step.DOWNLOADING]:
            success = await app_state.delete_pipeline()
            if not success:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            return {"message": "Pipeline cancelled and deleted", "action": "deleted"}

        elif step in [Step.AUDIO_ANALYSIS, Step.VAD_PREP, Step.VAD_ANALYSIS]:
            await pipeline.restart_at_step(RestartStep.SELECT_WORKFLOW)
            return {
                "message": "Processing cancelled, returned to workflow selection",
                "action": "restarted",
                "restart_step": RestartStep.SELECT_WORKFLOW.value,
            }

        elif step in intelligent_steps or (
            expected_step in intelligent_steps
            and (
                (step == Step.CONFIGURE_ASR and pipeline._intelligent_vosk_evidence is not None)
                or step == Step.REFERENCE_VALIDATION_RESULTS
                or (step == Step.SELECT_WORKFLOW and pipeline.reference_validation_origin == Step.SELECT_WORKFLOW)
            )
        ):
            # Intelligent detection is optional and sits on top of a completed
            # silence scan.  Cancelling it must preserve that scan and return
            # to its own form. The expected-step check also closes the narrow
            # race where the LLM finishes after the browser displays Cancel but
            # before this request reaches the backend.
            app_state.progress_dispatcher.increment_epoch()
            await pipeline.cancel_processing()
            if (
                step in {Step.REFERENCE_VALIDATION, Step.SELECT_WORKFLOW}
                and pipeline.reference_validation_origin == Step.SELECT_WORKFLOW
            ):
                pipeline._reference_vosk_results = []
                pipeline._notify_progress(
                    Step.SELECT_WORKFLOW,
                    0,
                    "Reference scan cancelled",
                )
                return {
                    "message": "Reference scan cancelled, returned to workflow selection",
                    "action": "restarted",
                    "restart_step": Step.SELECT_WORKFLOW.value,
                }

            await pipeline._transition_to_intelligent_chapter_detection()
            return {
                "message": "Intelligent chapter detection cancelled, returned to its configuration",
                "action": "restarted",
                "restart_step": Step.INTELLIGENT_CHAPTER_DETECTION.value,
            }

        elif step == Step.AUDIO_EXTRACTION:
            restart_step = RestartStep.SELECT_WORKFLOW if pipeline.is_realignment else RestartStep.CONFIGURE_ASR
            await pipeline.restart_at_step(restart_step)
            return {
                "message": f"Audio extraction cancelled, returned to {restart_step.value}",
                "action": "restarted",
                "restart_step": restart_step.value,
            }

        elif step in [Step.TRIMMING, Step.ASR_PROCESSING]:
            await pipeline.restart_at_step(RestartStep.CONFIGURE_ASR)
            return {
                "message": "Transcription process cancelled, returned to ASR configuration",
                "action": "restarted",
                "restart_step": RestartStep.CONFIGURE_ASR.value,
            }

        elif step in [Step.AI_CLEANUP, Step.PARTIAL_SCAN_PREP, Step.PARTIAL_AUDIO_ANALYSIS, Step.PARTIAL_VAD_ANALYSIS]:
            await pipeline.restart_at_step(RestartStep.CHAPTER_EDITING)
            return {
                "message": "Processing cancelled, returned to chapter editing",
                "action": "restarted",
                "restart_step": RestartStep.CHAPTER_EDITING.value,
            }

        else:
            await pipeline.restart_at_step(RestartStep.SELECT_WORKFLOW)
            return {
                "message": "Processing cancelled, returned to workflow selection",
                "action": "restarted",
                "restart_step": RestartStep.SELECT_WORKFLOW.value,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel current step: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/pipeline/asr-options")
async def update_asr_options(request: ASROptionsRequest):
    """Update ASR options for the pipeline"""
    try:
        from ...core.config import get_app_config, update_app_config

        # Update ASR options in config (persistent)
        app_config = get_app_config()
        app_config.asr_options.trim = request.trim
        app_config.asr_options.use_bias_words = request.use_bias_words
        app_config.asr_options.bias_words = request.bias_words
        app_config.asr_options.segment_length = request.segment_length

        success = update_app_config(app_config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update ASR options")

        return {
            "message": "ASR options updated successfully",
            "options": app_config.asr_options.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update ASR options for pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/asr-options")
async def get_asr_options():
    """Get ASR options for the current pipeline"""
    try:
        from ...core.config import get_app_config

        # Get ASR options from config (persistent)
        app_config = get_app_config()

        return {
            "options": app_config.asr_options.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ASR options for pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))
