"""Targeted Vosk evidence collection for silence-derived chapter candidates."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from app.core.system_info import get_worker_count
from app.models.enums import Step
from app.models.progress import ProgressCallback

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("ACHEW_VOSK_MODEL_PATH")
MODEL_NAME = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 2
GATE_BEFORE_SECONDS = 0.5
GATE_AFTER_SECONDS = 3.0
TERMS = [
    # Structural headings and common audiobook front/back matter.  This is a
    # constrained grammar, so every word we might want to pass to the LLM must
    # be listed explicitly here.
    "chapter",
    "section",
    "part",
    "letter",
    "book",
    "volume",
    "subsection",
    "paragraph",
    "page",
    "segment",
    "scene",
    "act",
    "episode",
    "introduction",
    "preface",
    "epilogue",
    "prologue",
    "foreword",
    "dedication",
    "acknowledgments",
    "afterword",
    "notes",
    "endnotes",
    "audible",
    # The small English model lacks "libri" and "librivox", but recognizes
    # the supported suffix when it is spoken as part of LibriVox credits.
    "vox",
    "recording",
    "summary",
    "previously",
    "preview",
    "epigraph",
    "recap",
    "appendix",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
]


def normalize_vosk_terms(values: Iterable[str]) -> List[str]:
    """Return unique, grammar-safe words in the order supplied by the user."""
    normalized: List[str] = []
    for value in values:
        # Vosk's grammar is word based; split pasted text and LLM suggestions
        # into individual words rather than passing punctuation or prose.
        for word in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(value).lower()):
            if word and word not in normalized:
                normalized.append(word)
    return normalized


class _VoskAPI(NamedTuple):
    model: Any
    recognizer: Any
    set_log_level: Callable[[int], None]


_vosk_api: Optional[_VoskAPI] = None
_model: Optional[Any] = None
_model_lock = threading.Lock()


@dataclass(frozen=True)
class VoskWord:
    word: str
    start: float
    end: float
    confidence: float
    utterance: int = 0
    utterance_start: Optional[float] = None
    utterance_end: Optional[float] = None


@dataclass(frozen=True)
class CandidateEvidence:
    timestamp: float
    words: List[VoskWord]


def is_vosk_available() -> bool:
    """Return whether Vosk-backed intelligent detection can run here."""
    return sys.platform != "darwin" and importlib.util.find_spec("vosk") is not None


def _get_vosk_api() -> _VoskAPI:
    """Import Vosk only when intelligent detection actually needs it."""
    global _vosk_api
    if _vosk_api is None:
        if sys.platform == "darwin":
            raise RuntimeError("Intelligent chapter detection is not available on native macOS")
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as e:
            raise RuntimeError("Vosk is required for intelligent chapter detection") from e
        _vosk_api = _VoskAPI(Model, KaldiRecognizer, SetLogLevel)
    return _vosk_api


def _get_model() -> Any:
    global _model
    with _model_lock:
        if _model is None:
            vosk = _get_vosk_api()
            if MODEL_PATH and not os.path.isdir(MODEL_PATH):
                raise RuntimeError(f"The configured Vosk model path does not exist: {MODEL_PATH}")
            vosk.set_log_level(-1)
            try:
                _model = vosk.model(MODEL_PATH) if MODEL_PATH else vosk.model(model_name=MODEL_NAME)
            except Exception as e:
                raise RuntimeError(
                    "Unable to load the Vosk model. Check the network connection and model cache permissions."
                ) from e
        return _model


def _merged_windows(anchors: Sequence[float], duration: float) -> List[Tuple[float, float]]:
    windows = sorted(
        (max(0.0, anchor - GATE_BEFORE_SECONDS), min(duration, anchor + GATE_AFTER_SECONDS)) for anchor in anchors
    )
    merged: List[Tuple[float, float]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class VoskCandidateService:
    """Run grammar-constrained recognition only where a silence suggests a boundary."""

    def __init__(
        self,
        progress_callback: ProgressCallback,
        running_processes: Optional[list] = None,
        process_lock: Optional[threading.Lock] = None,
        terms: Optional[Iterable[str]] = None,
    ) -> None:
        self.progress_callback = progress_callback
        self._running_processes = running_processes if running_processes is not None else []
        self._process_lock = process_lock or threading.Lock()
        self.terms = normalize_vosk_terms(terms if terms is not None else TERMS)

    def _notify(self, percent: float, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.progress_callback(Step.VOSK_ANALYSIS, percent, message, details or {})

    def _scan_window(
        self,
        audio_file: str,
        start: float,
        end: float,
        cancelled: threading.Event,
    ) -> List[VoskWord]:
        """Recognize one short window without materializing an audio file.

        Every worker gets its own recognizer while sharing the cached, read-only
        Vosk model.  The cancellation event prevents queued work from starting
        after the pipeline has stopped the scan.
        """
        if cancelled.is_set():
            return []

        recognizer = _get_vosk_api().recognizer(_get_model(), SAMPLE_RATE, json.dumps([*self.terms, "[unk]"]))
        recognizer.SetWords(True)
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.6f}",
                "-i",
                audio_file,
                "-t",
                f"{end - start:.6f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "s16le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with self._process_lock:
            self._running_processes.append(process)
        words: List[VoskWord] = []

        def collect(payload: str, utterance: int) -> None:
            rows = json.loads(payload).get("result", [])
            if not rows:
                return
            utterance_start = round(start + min(float(row["start"]) for row in rows), 3)
            utterance_end = round(start + max(float(row["end"]) for row in rows), 3)
            for row in rows:
                word = str(row.get("word", "")).lower()
                if word in self.terms:
                    words.append(
                        VoskWord(
                            word=word,
                            start=round(start + float(row["start"]), 3),
                            end=round(start + float(row["end"]), 3),
                            confidence=round(float(row.get("conf", 0.0)), 3),
                            utterance=utterance,
                            utterance_start=utterance_start,
                            utterance_end=utterance_end,
                        )
                    )

        try:
            if process.stdout is None:
                raise RuntimeError("ffmpeg did not expose decoded audio")
            utterance = 0
            while chunk := process.stdout.read(8000):
                if cancelled.is_set():
                    process.terminate()
                    return []
                if recognizer.AcceptWaveform(chunk):
                    collect(recognizer.Result(), utterance)
                    utterance += 1
            if cancelled.is_set():
                return []
            collect(recognizer.FinalResult(), utterance)
            if process.wait():
                stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                raise RuntimeError(f"Vosk audio decode failed: {stderr[:500]}")
            return words
        finally:
            with self._process_lock:
                if process in self._running_processes:
                    self._running_processes.remove(process)
            if process.poll() is None:
                process.terminate()

    def _scan(
        self,
        audio_file: str,
        duration: float,
        anchors: Sequence[float],
        cancelled: threading.Event,
    ) -> List[CandidateEvidence]:
        windows = _merged_windows(anchors, duration)
        if not windows:
            return [CandidateEvidence(timestamp=anchor, words=[]) for anchor in anchors]

        # Unlike chapter extraction, these ranges are usually sparse across a
        # long book.  Feeding them through the contiguous clip extractor would
        # make ffmpeg decode all of the gaps and write temporary files.  Instead
        # run direct, short ffmpeg-to-PCM windows concurrently.  The shared
        # worker budget is already CPU/memory-aware and is used throughout the
        # audio pipeline.
        worker_count = max(1, min(get_worker_count(), len(windows)))
        window_words: List[Optional[List[VoskWord]]] = [None] * len(windows)
        completed = 0
        word_count = 0

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vosk-window") as executor:
            futures = {
                executor.submit(self._scan_window, audio_file, start, end, cancelled): index
                for index, (start, end) in enumerate(windows)
            }
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    if cancelled.is_set():
                        break
                    result = future.result()
                    window_words[index] = result
                    completed += 1
                    word_count += len(result)
                    self._notify(
                        completed / len(windows) * 100,
                        f"Checking spoken headings… ({completed}/{len(windows)} regions)",
                        {
                            "regions_completed": completed,
                            "region_count": len(windows),
                            "word_count": word_count,
                            "worker_count": worker_count,
                        },
                    )
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise

        if cancelled.is_set():
            return []

        # Completion order is intentionally irrelevant. Reassemble results in
        # time order while preserving Vosk's internal utterance boundaries.
        return [
            CandidateEvidence(
                timestamp=anchor,
                words=[
                    word
                    for (start, end), result in zip(windows, window_words)
                    if start <= anchor <= end
                    for word in (result or [])
                    if anchor - GATE_BEFORE_SECONDS <= word.start <= anchor + GATE_AFTER_SECONDS
                ],
            )
            for anchor in anchors
        ]

    async def collect(self, audio_file: str, duration: float, anchors: Iterable[float]) -> List[CandidateEvidence]:
        anchor_list = list(anchors)
        self._notify(0, "Loading local Vosk model… This may take a while the first time.")
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        try:
            # Load (and, on first use, download) the shared model before starting
            # worker threads so only recognition work runs in parallel.
            await loop.run_in_executor(None, _get_model)
            result = await loop.run_in_executor(None, self._scan, audio_file, duration, anchor_list, cancelled)
        except asyncio.CancelledError:
            # The pipeline then terminates all registered ffmpeg children.  Set
            # this first so queued executor jobs do not start another decoder.
            cancelled.set()
            raise
        self._notify(100, f"Checked spoken headings for {len(result)} silence candidates")
        return result
