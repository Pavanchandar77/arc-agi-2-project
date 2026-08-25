"""Four-way Bond evaluation.

learned_model_gain = accuracy(bond_direct) - accuracy(base_direct)
hrps_substrate_gain = accuracy(bond_hrps) - accuracy(bond_direct)
"""

from src.hrps.bond import SYSTEMS, bond_deltas, probe_aabf363d, run_bond_eval


def named_gains(summaries: dict) -> dict:
    d = bond_deltas(summaries)
    d["hrps_substrate_gain"] = d["delta_substrate_bond_hrps_minus_bond_direct"]
    d["learned_model_gain_value"] = d["delta_train_bond_direct_minus_base_direct"]
    return d


__all__ = ["SYSTEMS", "bond_deltas", "named_gains", "probe_aabf363d", "run_bond_eval"]
