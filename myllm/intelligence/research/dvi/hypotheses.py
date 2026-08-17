"""
myllm.intelligence.research.dvi.hypotheses — Formal research hypotheses for DVI.
Contains NO unvalidated numerical claims. Formulates measurable, falsifiable experimental criteria.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ResearchHypothesis:
    hypothesis_id: str
    title: str
    formal_statement: str
    independent_variables: List[str]
    dependent_variables: List[str]
    falsification_criteria: str
    baseline_comparison: str


DVI_HYPOTHESES: Dict[str, ResearchHypothesis] = {
    "H1_CAPABILITY_PER_TOKEN": ResearchHypothesis(
        hypothesis_id="DVI-H1",
        title="Value-Weighted Pretraining Data Efficiency",
        formal_statement=(
            "Prioritizing pretraining examples with high loss gradient variance and low label noise "
            "reduces the total training token budget required to achieve equivalent validation loss "
            "compared to uniform random sampling."
        ),
        independent_variables=["Sampling distribution (Uniform vs DVI Value-Scored)", "Data mixture weights"],
        dependent_variables=["Validation perplexity", "Held-out cross-entropy loss", "Tokens consumed to target loss"],
        falsification_criteria="If the number of tokens to achieve target loss delta <= 2% or increases under DVI sampling.",
        baseline_comparison="Standard i.i.d. uniform random sampling over identical token corpus.",
    ),
    "H2_COMPUTE_PER_ACCURACY": ResearchHypothesis(
        hypothesis_id="DVI-H2",
        title="Adaptive Compute vs. Uniform Fixed Compute Inference",
        formal_statement=(
            "Routing queries adaptively across Fast, Think, and Tool paths achieves competitive accuracy "
            "with a lower aggregate inference FLOP consumption across a balanced evaluation benchmark."
        ),
        independent_variables=["Inference routing policy (Uniform Deep vs Adaptive Router)", "Compute budget caps"],
        dependent_variables=["Downstream task accuracy", "Total tokens generated", "Inference latency (ms)", "Total FLOPs"],
        falsification_criteria="If adaptive routing achieves lower accuracy at equivalent FLOPs or higher FLOPs at equal accuracy.",
        baseline_comparison="Uniform fixed multi-pass reasoning over all queries.",
    ),
}
