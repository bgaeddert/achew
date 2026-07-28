from app.services.llm_providers.base import CandidateTriageList, compact_candidate_evidence


def test_compact_candidate_evidence_keeps_only_independent_triage_signals():
    candidates = [
        {
            "candidate_id": "candidate-4",
            "timestamp_seconds": 123.67,
            "silence_seconds": 3.24,
            "first_word_offset_seconds": 0.283,
            "spoken_terms": "chapter four",
            "structural_words": [{"word": "chapter", "confidence": 1.0}],
        }
    ]

    assert compact_candidate_evidence(candidates) == [[124, 3.2, 0.3, "chapter four"]]


def test_compact_candidate_evidence_reads_legacy_debug_words():
    candidates = [
        {
            "timestamp_seconds": 10.0,
            "silence_seconds": 2.0,
            "structural_words": [{"word": "part"}, {"word": "one"}],
        }
    ]

    assert compact_candidate_evidence(candidates) == [[10, 2.0, 0.0, "part one"]]


def test_native_triage_response_is_an_accepted_id_list():
    assert CandidateTriageList.model_validate({"accepted_ids": [0, 2]}).accepted_ids == [0, 2]
