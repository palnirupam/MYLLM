"""
myllm.intelligence.retrieval.benchmark — A/B benchmark harness for retrieval vs non-retrieval.
Measures factuality, unsupported claim rates, latencies, and token expenditures.
"""

from typing import List, Dict, Any, Optional
import time
from myllm.intelligence.schemas import Document, VerificationStatus, RoutePath
from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.retrieval.bm25 import InMemoryBM25Retriever
from myllm.runtime.interfaces.base import InferenceRuntime


BENCHMARK_FACTUAL_DATA = [
    {
        "query": "Who developed the theory of general relativity and in what year was it published?",
        "ground_truth": "Albert Einstein developed general relativity in 1915.",
        "documents": [
            Document(
                doc_id="doc_relativity_01",
                content="General relativity is the geometric theory of gravitation published by Albert Einstein in 1915.",
                source="Physics History Archive",
                provenance_uri="https://archive.org/physics/relativity",
                language="en",
            )
        ]
    },
    {
        "query": "What is the boiling point of liquid nitrogen at standard atmospheric pressure?",
        "ground_truth": "The boiling point of liquid nitrogen is -195.79 °C (77 Kelvin).",
        "documents": [
            Document(
                doc_id="doc_nitrogen_01",
                content="Liquid nitrogen boils at 77.36 K (-195.79 °C or -320.4 °F) at atmospheric pressure.",
                source="Cryogenics Handbook",
                provenance_uri="https://cryo.org/nitrogen",
                language="en",
            )
        ]
    },
    {
        "query": "When was the James Webb Space Telescope launched into space?",
        "ground_truth": "The James Webb Space Telescope was launched on December 25, 2021.",
        "documents": [
            Document(
                doc_id="doc_jwst_01",
                content="The James Webb Space Telescope (JWST) was launched on 25 December 2021 on an Ariane 5 rocket.",
                source="NASA Space Missions",
                provenance_uri="https://nasa.gov/jwst",
                language="en",
            )
        ]
    },
]


def run_retrieval_ab_benchmark(runtime: InferenceRuntime) -> Dict[str, Any]:
    """
    Runs A/B comparative benchmark:
    Variant A: Retrieval Disabled (queries forced to FastPath baseline)
    Variant B: Retrieval Enabled (queries routed to RetrievePath + BM25)
    """
    retriever = InMemoryBM25Retriever()
    for item in BENCHMARK_FACTUAL_DATA:
        retriever.add_documents(item["documents"])

    # Variant A: Orchestrator without documents
    orchestrator_a = DhruvaOrchestrator(runtime=runtime, retriever=InMemoryBM25Retriever())
    # Variant B: Orchestrator with indexed documents
    orchestrator_b = DhruvaOrchestrator(runtime=runtime, retriever=retriever)

    results_a = []
    results_b = []

    for item in BENCHMARK_FACTUAL_DATA:
        query = item["query"]

        # Run A (Retrieval Disabled / Empty Store)
        start_a = time.time()
        res_a = orchestrator_a.answer(query)
        lat_a = (time.time() - start_a) * 1000.0

        results_a.append({
            "query": query,
            "status": res_a.verification_status.value,
            "confidence": res_a.confidence,
            "citations_count": len(res_a.evidence_citations),
            "latency_ms": lat_a,
            "tokens": res_a.telemetry.get("tokens_consumed", 0),
        })

        # Run B (Retrieval Enabled / Indexed Store)
        start_b = time.time()
        res_b = orchestrator_b.answer(query)
        lat_b = (time.time() - start_b) * 1000.0

        results_b.append({
            "query": query,
            "status": res_b.verification_status.value,
            "confidence": res_b.confidence,
            "citations_count": len(res_b.evidence_citations),
            "citations": res_b.evidence_citations,
            "latency_ms": lat_b,
            "tokens": res_b.telemetry.get("tokens_consumed", 0),
        })

    # Summary metrics
    a_pass_rate = sum(1 for r in results_a if r["status"] == "PASS") / len(results_a)
    a_unverified_rate = sum(1 for r in results_a if r["status"] in ("UNVERIFIED", "ABSTAIN")) / len(results_a)
    a_avg_lat = sum(r["latency_ms"] for r in results_a) / len(results_a)
    a_avg_tokens = sum(r["tokens"] for r in results_a) / len(results_a)

    b_pass_rate = sum(1 for r in results_b if r["status"] == "PASS") / len(results_b)
    b_unverified_rate = sum(1 for r in results_b if r["status"] in ("UNVERIFIED", "ABSTAIN")) / len(results_b)
    b_avg_lat = sum(r["latency_ms"] for r in results_b) / len(results_b)
    b_avg_tokens = sum(r["tokens"] for r in results_b) / len(results_b)

    return {
        "variant_a_retrieval_disabled": {
            "verified_pass_rate": a_pass_rate,
            "unverified_or_abstain_rate": a_unverified_rate,
            "average_latency_ms": round(a_avg_lat, 2),
            "average_tokens": round(a_avg_tokens, 1),
            "results": results_a,
        },
        "variant_b_retrieval_enabled": {
            "verified_pass_rate": b_pass_rate,
            "unverified_or_abstain_rate": b_unverified_rate,
            "average_latency_ms": round(b_avg_lat, 2),
            "average_tokens": round(b_avg_tokens, 1),
            "results": results_b,
        },
    }
