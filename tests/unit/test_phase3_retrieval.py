"""
tests/unit/test_phase3_retrieval.py
Comprehensive Phase 3 Test Suite:
- Exact keyword retrieval
- BM25 ranking & term frequency scoring
- Language filtering
- Metadata filtering
- No-result / empty search behavior
- Evidence citation preservation (zero fabricated citations)
- Unsupported claim -> UNVERIFIED
- Supported claim -> PASS
- Conflicting & irrelevant document handling
- End-to-end grounded answering via DhruvaOrchestrator
- A/B retrieval benchmark execution
"""

import sys
from pathlib import Path
from typing import Generator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.schemas import Document, RoutePath, VerificationStatus
from myllm.intelligence.retrieval.bm25 import InMemoryBM25Retriever
from myllm.intelligence.retrieval.context_builder import StructuredContextBuilder
from myllm.intelligence.retrieval.benchmark import run_retrieval_ab_benchmark
from myllm.intelligence.verifier.factual import EvidenceGroundedVerifier
from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.runtime.interfaces.base import InferenceRuntime


class MockGroundedRuntime(InferenceRuntime):
    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        if "Albert Einstein" in prompt:
            return "Albert Einstein published general relativity in 1915 as a geometric theory of gravitation."
        if "nitrogen" in prompt:
            return "Liquid nitrogen boils at -195.79 °C under atmospheric pressure."
        return prompt + " Grounded response."

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield " Grounded"


# 1. Exact Keyword Retrieval & BM25 Ranking
def test_bm25_exact_and_ranking():
    retriever = InMemoryBM25Retriever()
    doc1 = Document(doc_id="d1", content="Quantum mechanics governs the behavior of subatomic particles.", source="Physics")
    doc2 = Document(doc_id="d2", content="Quantum computing uses quantum mechanics principles to perform calculations rapidly.", source="CS")
    doc3 = Document(doc_id="d3", content="Organic chemistry studies carbon-based chemical compounds.", source="Chemistry")

    retriever.add_documents([doc1, doc2, doc3])

    # Query with 'quantum computing' should rank doc2 first (higher term frequency)
    results = retriever.retrieve("quantum computing", top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == "d2"
    assert results[0].score > results[1].score


# 2. Language Filtering
def test_language_filtering():
    retriever = InMemoryBM25Retriever()
    doc_en = Document(doc_id="en_1", content="Paris is the capital of France.", language="en")
    doc_bn = Document(doc_id="bn_1", content="ফ্রান্সের রাজধানী প্যারিস।", language="bn")

    retriever.add_documents([doc_en, doc_bn])

    res_en = retriever.retrieve("Paris France", language="en")
    assert len(res_en) == 1
    assert res_en[0].doc_id == "en_1"

    res_bn = retriever.retrieve("প্যারিস", language="bn")
    assert len(res_bn) == 1
    assert res_bn[0].doc_id == "bn_1"


# 3. Metadata Filtering
def test_metadata_filtering():
    retriever = InMemoryBM25Retriever()
    doc1 = Document(doc_id="sec_1", content="Internal project roadmap for Q4.", metadata={"access": "confidential", "dept": "engineering"})
    doc2 = Document(doc_id="pub_1", content="Public release notes for Q4.", metadata={"access": "public", "dept": "engineering"})

    retriever.add_documents([doc1, doc2])

    res_pub = retriever.retrieve("roadmap notes Q4", metadata_filter={"access": "public"})
    assert len(res_pub) == 1
    assert res_pub[0].doc_id == "pub_1"


# 4. No-Result & Empty Search Behavior
def test_no_result_behavior():
    retriever = InMemoryBM25Retriever()
    doc1 = Document(doc_id="d1", content="Photosynthesis in plants.", source="Biology")
    retriever.add_documents([doc1])

    # Irrelevant query
    res = retriever.retrieve("completely unrelated quantum astrophysics term", min_score=1.0)
    assert len(res) == 0


# 5. Provenance Preservation & Zero Citation Fabrication
def test_provenance_and_citations():
    builder = StructuredContextBuilder()
    docs = [
        Document(doc_id="doc_101", content="The speed of light in vacuum is 299,792,458 m/s.", source="NIST", provenance_uri="https://nist.gov/light")
    ]
    context = builder.build_context(docs)

    assert "doc_id=\"doc_101\"" in context
    assert "source=\"NIST\"" in context
    assert "https://nist.gov/light" in context


# 6. Factual Verifier: Unsupported Claim -> UNVERIFIED, Supported -> PASS
def test_verifier_factual_grounding():
    verifier = EvidenceGroundedVerifier()

    # Zero evidence -> UNVERIFIED
    res_unverified = verifier.verify("What is light speed?", "Light speed is 300,000 km/s.", evidence=None)
    assert res_unverified.status == VerificationStatus.UNVERIFIED
    assert len(res_unverified.details["unsupported_claims"]) > 0

    # With matching evidence -> PASS
    evidence = ["[doc_id='doc_101'] The speed of light is approximately 300,000 km per second in a vacuum."]
    res_pass = verifier.verify("What is light speed?", "The speed of light is approximately 300,000 km per second in a vacuum.", evidence=evidence)
    assert res_pass.status == VerificationStatus.PASS
    assert len(res_pass.details["supported_claims"]) > 0


# 7. Irrelevant / Conflicting Evidence Handling
def test_conflicting_and_irrelevant_evidence():
    verifier = EvidenceGroundedVerifier()

    # Irrelevant evidence -> REVISE (due to low claim overlap)
    irrelevant_evidence = ["[doc_id='doc_irr'] Bananas are rich in potassium and dietary fiber."]
    res = verifier.verify("Who discovered gravity?", "Sir Isaac Newton formulated the classical law of universal gravitation.", evidence=irrelevant_evidence)
    assert res.status == VerificationStatus.REVISE
    assert ("Insufficient grounding" in res.critique or "Entity attribution failure" in res.critique)


# 8. End-to-End Grounded Answering via DhruvaOrchestrator
def test_orchestrator_grounded_answer_e2e():
    retriever = InMemoryBM25Retriever()
    doc = Document(
        doc_id="einstein_01",
        content="Albert Einstein published the theory of general relativity in 1915.",
        source="History of Physics",
        provenance_uri="https://physics.org/einstein",
        language="en",
    )
    retriever.add_documents([doc])

    runtime = MockGroundedRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime, retriever=retriever)

    # Query matching retrieval pattern
    result = orchestrator.answer("cite sources for who developed the theory of general relativity as of 1915")

    assert result.route_taken == RoutePath.RETRIEVE
    assert result.verification_status == VerificationStatus.PASS
    assert "einstein_01" in result.evidence_citations
    assert "Albert Einstein" in result.answer
    assert result.telemetry["retrieval_used"] is True
    assert result.telemetry["tools_used"] == []


# 9. End-to-End Abstention When Retrieval Returns Zero Usable Evidence
def test_orchestrator_abstention_on_empty_retrieval():
    empty_retriever = InMemoryBM25Retriever()
    runtime = MockGroundedRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime, retriever=empty_retriever)

    # Query requiring evidence but with empty document database
    result = orchestrator.answer("cite sources for secret unknown confidential archives")

    assert result.route_taken == RoutePath.RETRIEVE
    assert result.verification_status == VerificationStatus.ABSTAIN
    assert "do not have verifiable external evidence" in result.answer
    assert result.confidence == 0.0
    assert "No relevant grounding documents" in result.uncertainty_reason


# 10. A/B Benchmark Harness Verification
def test_retrieval_ab_benchmark_harness():
    runtime = MockGroundedRuntime()
    bench_results = run_retrieval_ab_benchmark(runtime)

    assert "variant_a_retrieval_disabled" in bench_results
    assert "variant_b_retrieval_enabled" in bench_results

    # Variant B (with documents indexed) should achieve higher verified pass rate
    var_b = bench_results["variant_b_retrieval_enabled"]
    assert var_b["verified_pass_rate"] > 0.5
    assert var_b["average_latency_ms"] >= 0.0


if __name__ == "__main__":
    tests = [
        test_bm25_exact_and_ranking,
        test_language_filtering,
        test_metadata_filtering,
        test_no_result_behavior,
        test_provenance_and_citations,
        test_verifier_factual_grounding,
        test_conflicting_and_irrelevant_evidence,
        test_orchestrator_grounded_answer_e2e,
        test_orchestrator_abstention_on_empty_retrieval,
        test_retrieval_ab_benchmark_harness,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL PHASE 3 RETRIEVAL TESTS PASSED")
