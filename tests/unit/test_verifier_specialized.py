"""
tests/unit/test_verifier_specialized.py
Verifies specialized verifiers:
- StructuralVerifier (empty, repetition loops)
- MathematicalVerifier (equation validation & contradiction detection)
- CodeExecutionVerifier (syntax validity & errors)
- EvidenceGroundedVerifier (strictly UNVERIFIED without evidence; PASS with evidence)
- CompositeVerifier (end-to-end multi-check aggregation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.verifier.structural import StructuralVerifier
from myllm.intelligence.verifier.mathematical import MathematicalVerifier
from myllm.intelligence.verifier.code import CodeExecutionVerifier
from myllm.intelligence.verifier.factual import EvidenceGroundedVerifier
from myllm.intelligence.verifier.composite import CompositeVerifier
from myllm.intelligence.schemas import VerificationStatus


def test_structural_verifier():
    verifier = StructuralVerifier(max_repetition_threshold=0.25)

    # Empty string -> REVISE
    empty_res = verifier.verify("query", "   ")
    assert empty_res.status == VerificationStatus.REVISE
    assert "empty" in empty_res.critique

    # Degenerative loop -> REVISE
    loop_text = "the dog barked the dog barked the dog barked the dog barked the dog barked"
    loop_res = verifier.verify("query", loop_text)
    assert loop_res.status == VerificationStatus.REVISE
    assert "repetitive loops" in loop_res.critique

    # Normal text -> PASS
    clean_text = "Dhruva is an intelligent model designed for speed, factuality, and reasoning."
    clean_res = verifier.verify("query", clean_text)
    assert clean_res.status == VerificationStatus.PASS


def test_mathematical_verifier():
    verifier = MathematicalVerifier()

    # Correct equation -> PASS
    correct_ans = "The result of 25 * 4 = 100 as calculated."
    res = verifier.verify("25 * 4", correct_ans)
    assert res.status == VerificationStatus.PASS

    # Contradictory equation -> REVISE
    wrong_ans = "The result of 25 * 4 = 105 as calculated."
    wrong_res = verifier.verify("25 * 4", wrong_ans)
    assert wrong_res.status == VerificationStatus.REVISE
    assert "contradictions detected" in wrong_res.critique

    # No math in answer -> UNVERIFIED (neutral)
    no_math = "Dhruva is a fast language model."
    neutral_res = verifier.verify("who is dhruva", no_math)
    assert neutral_res.status == VerificationStatus.UNVERIFIED


def test_code_execution_verifier():
    verifier = CodeExecutionVerifier()

    # Valid Python code block -> PASS
    valid_code = "```python\ndef add(a, b):\n    return a + b\n```"
    res = verifier.verify("write add function", valid_code)
    assert res.status == VerificationStatus.PASS

    # Syntax error in code block -> REVISE
    bad_code = "```python\ndef add(a, b\n    return a + b\n```"
    bad_res = verifier.verify("write add function", bad_code)
    assert bad_res.status == VerificationStatus.REVISE
    assert "SyntaxError" in bad_res.critique


def test_evidence_grounded_verifier_strict_unverified():
    verifier = EvidenceGroundedVerifier()

    # CRITICAL TEST: Unsupported factual claim with zero evidence MUST be UNVERIFIED, NEVER PASS
    res_no_evidence = verifier.verify("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928.", evidence=None)
    assert res_no_evidence.status == VerificationStatus.UNVERIFIED
    assert "No external grounded evidence" in res_no_evidence.critique

    # With matching evidence -> PASS
    evidence = ["Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928."]
    res_with_evidence = verifier.verify("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928.", evidence=evidence)
    assert res_with_evidence.status == VerificationStatus.PASS


def test_composite_verifier():
    composite = CompositeVerifier()

    # Normal factual query with evidence -> PASS
    ans = "The capital of France is Paris. 2 + 2 = 4."
    evidence = ["Paris is the capital and most populous city of France."]
    res = composite.verify("capital of France", ans, evidence=evidence)
    assert res.status == VerificationStatus.PASS
    assert res.details["structural"] == "PASS"
    assert res.details["mathematical"] == "PASS"


if __name__ == "__main__":
    tests = [
        test_structural_verifier,
        test_mathematical_verifier,
        test_code_execution_verifier,
        test_evidence_grounded_verifier_strict_unverified,
        test_composite_verifier,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL SPECIALIZED VERIFIER TESTS PASSED")
