"""Bond provenance: hashes, licenses, foundation ids, configs."""

from src.hrps.bond import bond_manifest, code_revision
from src.hrps.identity import PUBLIC_NAME, QWEN_LICENSE_NOTE
from src.hrps.package import LAYOUT, write_bond_package, write_remote_train_bundle

MODULE_MAP = {
    "hrps_core": "src.hrps.core",
    "arc_adapter": "src.hrps.arc_adapter",
    "bond_model": "src.hrps.identity + src.hrps.backend",
    "bond_schema": "src.hrps.schema",
    "bond_overseer": "src.hrps.bond_overseer",
    "bond_memory": "src.hrps.bond_memory",
    "bond_tools": "src.hrps.bond_tools",
    "bond_episode": "src.hrps.episodes",
    "bond_train": "src.hrps.bond_train",
    "bond_eval": "src.hrps.bond_eval",
    "bond_manifest": "src.hrps.bond_manifest",
    "bond_l1_language": "src.hrps.language",
    "bond_role_ops": "src.hrps.role_ops",
    "bond_guided_search": "src.hrps.guided_search",
    "bond_engine": "src.hrps.bond_engine",
    "bond_synthesizer": "src.hrps.synthesize",
    "bond_verifier_reward": "src.hrps.bond_reward",
    "bond_curriculum": "src.hrps.bond_curriculum",
    "executor": "src.hrps.dsl.replay",
    "verifier": "src.hrps.residual.joint_residual",
    "public_name": PUBLIC_NAME,
    "generality": "interface-only; ARC is the first proof domain, not an assumed result",
}

__all__ = [
    "LAYOUT",
    "MODULE_MAP",
    "PUBLIC_NAME",
    "QWEN_LICENSE_NOTE",
    "bond_manifest",
    "code_revision",
    "write_bond_package",
    "write_remote_train_bundle",
]
