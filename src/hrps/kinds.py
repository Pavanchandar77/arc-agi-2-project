"""Scientific labels for every HRPS component.

Allowed labels:
  exact                — total, deterministic, no approximation
  sound_incomplete     — never unsound; may miss equivalences or solutions
  heuristic            — useful bias; not a proof
  learned              — fitted from data (Phase H/I only)

Never call a heuristic bound admissible.
"""

from __future__ import annotations

from enum import Enum


class Kind(str, Enum):
    EXACT = "exact"
    SOUND_INCOMPLETE = "sound_but_incomplete"
    HEURISTIC = "heuristic"
    LEARNED = "learned"


# Frozen Phase-1 inventory. H is the next allowed language change; I (Qwen)
# stays blocked until a language change is measured with the A/F/G ladder.
COMPONENT_KIND: dict[str, Kind] = {
    "grid_type": Kind.EXACT,
    "task_loader": Kind.EXACT,
    "pixel_equality": Kind.EXACT,
    "executor": Kind.EXACT,
    "program_replay": Kind.EXACT,
    "program_serialization": Kind.EXACT,
    "description_length": Kind.EXACT,
    "joint_pixel_residual": Kind.EXACT,
    "joint_demonstration_verifier": Kind.EXACT,
    "connected_components_4": Kind.EXACT,
    "connected_components_8": Kind.EXACT,
    "color_agnostic_components": Kind.EXACT,
    "object_residual": Kind.SOUND_INCOMPLETE,
    "relation_residual": Kind.SOUND_INCOMPLETE,
    "shape_control_residual": Kind.EXACT,
    "multi_representation_bank": Kind.SOUND_INCOMPLETE,
    "colormap_generator": Kind.EXACT,
    "residual_operator_prioritization": Kind.HEURISTIC,
    "continuation_signature": Kind.SOUND_INCOMPLETE,
    "signature_dedup": Kind.SOUND_INCOMPLETE,
    "admissible_remaining_cost_bound": Kind.EXACT,
    "search_frontier_order": Kind.HEURISTIC,
    "failure_taxonomy": Kind.HEURISTIC,
    "abstraction_library": Kind.LEARNED,  # which macros are named, from training traces
    "abstraction_execution": Kind.EXACT,  # macro body is exact DSL replay
    "qwen_proposals": Kind.LEARNED,  # blocked until H is measured
    "hrps_environment": Kind.EXACT,  # inspect / apply / residual / commit
    "gold_free_constraint_feedback": Kind.EXACT,  # train residuals + test-input evidence only
    "open_model_reasoner": Kind.LEARNED,  # frozen open model; not the verifier
    "bond_episode": Kind.EXACT,  # env replay of actions/residuals; selection is training-only
    "bond_adapter": Kind.LEARNED,  # LoRA/QLoRA parameters from HRPS episodes
    "bond_inference_controller": Kind.EXACT,  # same HRPS action loop as elevation M2
}


def kind_of(component: str) -> Kind:
    try:
        return COMPONENT_KIND[component]
    except KeyError as exc:
        raise KeyError(f"Unlabeled component {component!r}; refuse to ship unlabeled mechanisms") from exc
