# MyLLM Specification: Evaluation & Benchmark Contract

- **Document ID**: `SPEC-0013`
- **Status**: Stable / Phase 0
- **Version**: `1.0.0`
- **Domain**: Model Evaluation, Quality Verification & Release Gating
- **Author**: MyLLM Core Architecture Team
- **Date**: 2026-08-16

---

## 1. Overview & Core Philosophy

The `EvaluationContract` standardizes the execution, metrics capture, and release gating criteria for model checkpoints and production candidates.

### Permanent Regression Suite Requirement:
No model artifact may be promoted to production or published to the official registry without passing the **Permanent Regression Suite**. This suite contains golden test vectors across all supported capability domains. Any regression greater than 0.5% on safety, tool execution syntax, or core reasoning blocks deployment automatically.

---

## 2. Evaluation Categories Taxonomy

| Category Code | Description | Standard Reference Datasets |
| :--- | :--- | :--- |
| `general_knowledge` | Multi-subject factual recall and trivia | MMLU-Redux, ARC-Challenge, CommonsenseQA |
| `reasoning` | Multi-step logical, deductive, and abductive reasoning | GSM8K, MATH-500, GPQA-Diamond, BigBench-Hard |
| `mathematics` | Symbolic, algebraic, and Olympiad mathematics | OlympiadBench, AIME-2024, Minerva Math |
| `coding` | Code generation, bug fixing, test synthesis | HumanEval, SWE-bench Lite, MBPP-Plus, MultiPL-E |
| `bengali` | Bengali NLU, translation, reasoning, sentiment | IndicGLUE, Bengali-MMLU, ProthomAlo-Sum, BanglaRQA |
| `english` | Complex English syntactic & semantic comprehension | SuperGLUE, Lambada, Winogrande |
| `multilingual` | Cross-lingual zero-shot transfer & multilingual QA | Flores-200, MGSM, TyDi QA |
| `long_context` | Needle-in-a-haystack, multi-hop retrieval over 128k | RULER, L-Eval, BABILong |
| `tool_usage` | Schema validation, parallel tool calls, multi-turn APIs | Berkeley Function Calling Leaderboard (BFCL), ToolBench |
| `safety` | Jailbreak defense, toxicity refusal, PII non-leakage | Do-Not-Answer, AdvGLUE, RealToxicityPrompts, XSTest |
| `regression` | Golden platform test vectors & bug reproducers | MyLLM Internal Golden Release Suite |

---

## 3. Protobuf Specification (`myllm/evaluation/v1/evaluation.proto`)

```protobuf
syntax = "proto3";

package myllm.evaluation.v1;

import "google/protobuf/timestamp.proto";

option go_package = "github.com/myllm/core/gen/go/evaluation/v1;evaluationv1";
option java_package = "ai.myllm.evaluation.v1";

enum EvaluationCategory {
  EVAL_CATEGORY_UNSPECIFIED = 0;
  EVAL_CATEGORY_GENERAL_KNOWLEDGE = 1;
  EVAL_CATEGORY_REASONING = 2;
  EVAL_CATEGORY_MATHEMATICS = 3;
  EVAL_CATEGORY_CODING = 4;
  EVAL_CATEGORY_BENGALI = 5;
  EVAL_CATEGORY_ENGLISH = 6;
  EVAL_CATEGORY_MULTILINGUAL = 7;
  EVAL_CATEGORY_LONG_CONTEXT = 8;
  EVAL_CATEGORY_TOOL_USAGE = 9;
  EVAL_CATEGORY_SAFETY = 10;
  EVAL_CATEGORY_REGRESSION = 11;
}

message Benchmark {
  string benchmark_id = 1;
  string name = 2;
  string description = 3;
  EvaluationCategory category = 4;
  string canonical_citation = 5;
  string license_spdx = 6;
}

message BenchmarkVersion {
  string benchmark_version_id = 1;
  string benchmark_id = 2;
  string version = 3;
  string dataset_manifest_hash = 4;
  string prompt_template_hash = 5;
  string scoring_harness_commit = 6;
  uint32 num_samples = 7;
}

message Metric {
  string metric_name = 1; // e.g. "pass@1", "exact_match", "bleu_score", "refusal_rate"
  double value = 2;
  double standard_error = 3;
  double confidence_interval_95_low = 4;
  double confidence_interval_95_high = 5;
  string unit = 6; // "percentage", "score_0_to_100", "bits_per_byte"
}

message PerSampleOutput {
  string sample_id = 1;
  string prompt = 2;
  string model_output = 3;
  string reference_ground_truth = 4;
  bool is_correct = 5;
  double score = 6;
  string execution_trace = 7;
}

message EvaluationRun {
  string run_id = 1;
  string model_contract_ref = 2;
  string benchmark_version_id = 3;
  string evaluation_harness_version = 4;
  map<string, string> sampling_parameters = 5; // e.g. temperature="0.0", top_p="1.0"
  google.protobuf.Timestamp started_at = 6;
  google.protobuf.Timestamp completed_at = 7;
  string executed_on_hardware = 8;
}

message EvaluationResult {
  string result_id = 1;
  string evaluation_run_id = 2;
  string model_id = 3;
  EvaluationCategory category = 4;
  repeated Metric metrics = 5;
  repeated PerSampleOutput sample_outputs = 6;
  bool passed_regression_gate = 7;
  string gate_evaluation_summary = 8;
}
```

---

## 4. JSON Schema Representation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://specs.myllm.ai/v1/evaluation-contract.json",
  "title": "EvaluationResult",
  "type": "object",
  "required": [
    "result_id",
    "evaluation_run_id",
    "model_id",
    "category",
    "metrics",
    "passed_regression_gate"
  ],
  "properties": {
    "result_id": { "type": "string" },
    "evaluation_run_id": { "type": "string" },
    "model_id": { "type": "string" },
    "category": {
      "type": "string",
      "enum": [
        "EVAL_CATEGORY_GENERAL_KNOWLEDGE",
        "EVAL_CATEGORY_REASONING",
        "EVAL_CATEGORY_MATHEMATICS",
        "EVAL_CATEGORY_CODING",
        "EVAL_CATEGORY_BENGALI",
        "EVAL_CATEGORY_ENGLISH",
        "EVAL_CATEGORY_MULTILINGUAL",
        "EVAL_CATEGORY_LONG_CONTEXT",
        "EVAL_CATEGORY_TOOL_USAGE",
        "EVAL_CATEGORY_SAFETY",
        "EVAL_CATEGORY_REGRESSION"
      ]
    },
    "metrics": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["metric_name", "value"],
        "properties": {
          "metric_name": { "type": "string" },
          "value": { "type": "number" },
          "standard_error": { "type": "number" },
          "unit": { "type": "string" }
        }
      }
    },
    "passed_regression_gate": { "type": "boolean" },
    "gate_evaluation_summary": { "type": "string" }
  }
}
```

---

## 5. Complete JSON Example (Bengali Evaluation Result)

```json
{
  "result_id": "eval_res_2026_08_bn_0981",
  "evaluation_run_id": "run_eval_20260816_bengali_7b_01",
  "model_id": "myllm/bengali-reasoning-7b:1.0.0",
  "category": "EVAL_CATEGORY_BENGALI",
  "metrics": [
    {
      "metric_name": "indic_glue_overall_accuracy",
      "value": 88.42,
      "standard_error": 0.32,
      "confidence_interval_95_low": 87.79,
      "confidence_interval_95_high": 89.05,
      "unit": "percentage"
    },
    {
      "metric_name": "bangla_rqa_exact_match",
      "value": 81.15,
      "standard_error": 0.45,
      "confidence_interval_95_low": 80.26,
      "confidence_interval_95_high": 82.04,
      "unit": "percentage"
    }
  ],
  "passed_regression_gate": true,
  "gate_evaluation_summary": "Passed all 1,200 Bengali NLU golden regression vectors. Exceeded previous baseline (85.2%) by +3.22%."
}
```
