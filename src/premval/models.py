"""Display metadata for the generative models shown in the UI.

Maps internal samples-cache model keys (e.g. `esmflow_md_distilled`) to a
human display name, the source publication URL, and a one-line description.
The leaderboard uses these to render proper-cased, linked model names with a
hover tooltip instead of the raw cache key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Contamination = Literal["held_out", "weak_holdout", "uncertain"]

# AlphaFlow and ESMFlow are introduced in the same paper.
_FLOW_PAPER = "https://arxiv.org/abs/2402.04845"


_FLOW_CITATION = (
    "Jing, B., Berger, B., & Jaakkola, T. (2024). AlphaFold Meets Flow Matching "
    "for Generating Protein Ensembles. International Conference on Machine "
    "Learning (ICML). arXiv:2402.04845."
)


@dataclass(frozen=True)
class ModelInfo:
    """How a model is presented in the UI.

    Attributes:
        name: Human-facing display name (e.g. "ESMFlow-MD (distilled)").
        url: Source publication URL, or None if unknown.
        description: One-line summary shown as a hover tooltip.
        citation: Full bibliographic citation for the References section, or
            None if unknown.
        contamination: Strength of the held-out guarantee for the ATLAS
            evaluation: `held_out` (never trained on ATLAS/MD),
            `weak_holdout` (trained on ATLAS train, scored on held-out test,
            but the split is temporal only), or `uncertain`.
        contamination_basis: One-line evidence for the contamination label,
            shown as a hover tooltip on the leaderboard badge.
    """

    name: str
    url: str | None
    description: str | None
    citation: str | None
    contamination: Contamination
    contamination_basis: str


def _flow_family() -> dict[str, ModelInfo]:
    """Registry entries for the AlphaFlow/ESMFlow variants (base/distilled x MD/PDB)."""
    backbones = {"alphaflow": ("AlphaFlow", "AlphaFold"), "esmflow": ("ESMFlow", "ESMFold")}
    # Per tier: display label, training corpus, and the ATLAS held-out label.
    # The MD variants fine-tune on the ATLAS train split and are scored on the
    # held-out test split, but the split is temporal only (weak guarantee); the
    # PDB variants never see ATLAS/MD at all.
    tiers: dict[str, tuple[str, str, Contamination, str]] = {
        "md": (
            "MD",
            "ATLAS MD trajectories",
            "weak_holdout",
            "Trained on the ATLAS train split; scored on the held-out test "
            "split. Split is temporal only (no sequence-homology filter), so "
            "test homologs may resemble training data.",
        ),
        "pdb": (
            "PDB",
            "PDB structures",
            "held_out",
            "Trained on PDB structures only; no MD/ATLAS in training.",
        ),
    }
    out: dict[str, ModelInfo] = {}
    for fam, (label, backbone) in backbones.items():
        for tier, (tier_label, corpus, contamination, basis) in tiers.items():
            for distilled in (False, True):
                key = f"{fam}_{tier}_{'distilled' if distilled else 'base'}"
                name = f"{label}-{tier_label}" + (" (distilled)" if distilled else "")
                desc = (
                    f"{backbone} fine-tuned with flow matching to sample structural "
                    f"ensembles, trained on {corpus} (Jing et al., ICML 2024)"
                    + ("; distilled for fewer sampling steps." if distilled else ".")
                )
                out[key] = ModelInfo(
                    name=name,
                    url=_FLOW_PAPER,
                    description=desc,
                    citation=_FLOW_CITATION,
                    contamination=contamination,
                    contamination_basis=basis,
                )
    return out


MODEL_REGISTRY: dict[str, ModelInfo] = {
    **_flow_family(),
    "bioemu": ModelInfo(
        name="BioEmu",
        url="https://doi.org/10.1101/2024.12.05.626885",
        description=(
            "Generative deep-learning emulator of protein equilibrium "
            "ensembles (Lewis et al., 2024)."
        ),
        citation=(
            "Lewis, S., et al. (2024). Scalable emulation of protein equilibrium "
            "ensembles with generative deep learning. bioRxiv 2024.12.05.626885."
        ),
        contamination="held_out",
        contamination_basis=(
            "Not trained on ATLAS; training filtered to <40% sequence identity "
            "to test proteins (Lewis et al., 2024)."
        ),
    ),
    "esmdiff": ModelInfo(
        name="ESMDiff",
        url="https://arxiv.org/abs/2410.18403",
        description=(
            "ESM3 fine-tuned with masked diffusion to sample protein "
            "conformational ensembles by decoding structure tokens (Lu et al., "
            "ICLR 2025)."
        ),
        citation=(
            "Lu, J., Chen, X., Lu, S. Z., Shi, C., Guo, H., Bengio, Y., & Tang, J. "
            "(2025). Structure Language Models for Protein Conformation "
            "Generation. International Conference on Learning Representations "
            "(ICLR). arXiv:2410.18403."
        ),
        contamination="uncertain",
        contamination_basis=(
            "Fine-tunes gated ESM3; the relationship of its structure-token "
            "training corpus to ATLAS is not established."
        ),
    ),
}


def model_info(key: str) -> ModelInfo:
    """Display metadata for `key`, falling back to the raw key (no link) if unknown."""
    return MODEL_REGISTRY.get(
        key,
        ModelInfo(
            name=key,
            url=None,
            description=None,
            citation=None,
            contamination="uncertain",
            contamination_basis="Training data relative to ATLAS not established.",
        ),
    )
