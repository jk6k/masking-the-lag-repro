"""Shared FULLER phase1 variant registry and normalization helpers."""

from __future__ import annotations

import copy
from typing import Any

SWITCH_KEYS = ("meso", "flow", "det", "sparse", "phy")
INTERNAL_EXPERIMENT_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")

_FULLER_VARIANT_DEFAULTS: dict[str, dict[str, Any]] = {
    "E0": {
        "variant_id": "ASTRA",
        "public_module_stack": ["ASTRA"],
    },
    "E1": {
        "variant_id": "MESO",
        "public_module_stack": ["ASTRA", "MESO"],
    },
    "E2": {
        "variant_id": "HOPS",
        "public_module_stack": ["ASTRA", "HOPS"],
    },
    "E3": {
        "variant_id": "DET",
        "public_module_stack": ["ASTRA", "DET"],
    },
    "E4": {
        "variant_id": "SPARSE",
        "public_module_stack": ["ASTRA", "SPARSE"],
    },
    "E5": {
        "variant_id": "PHY",
        "public_module_stack": ["ASTRA", "PHY"],
    },
    "E6": {
        "variant_id": "FULLER",
        "public_module_stack": ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "PHY"],
    },
}


def normalize_switches(raw: dict[str, Any] | None) -> dict[str, bool]:
    payload = raw or {}
    return {key: bool(payload.get(key, False)) for key in SWITCH_KEYS}


def default_variant_descriptor_for_experiment(experiment_id: str) -> dict[str, Any]:
    normalized = str(experiment_id or "").strip().upper()
    default = _FULLER_VARIANT_DEFAULTS.get(normalized, {})
    return {
        "internal_experiment_id": normalized,
        "variant_id": default.get("variant_id") or normalized,
        "public_module_stack": list(default.get("public_module_stack") or [normalized]),
    }


def default_fuller_phase1_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "ASTRA",
            "internal_experiment_id": "E0",
            "lane_label": "astra_anchor",
            "mechanism_focus": "astra",
            "public_module_stack": ["ASTRA"],
            "config_stub": "astra",
            "switches": {"meso": False, "flow": False, "det": False, "sparse": False, "phy": False},
            "default_module_cfg": {},
        },
        {
            "variant_id": "MESO",
            "internal_experiment_id": "E1",
            "lane_label": "meso_module",
            "mechanism_focus": "meso",
            "public_module_stack": ["ASTRA", "MESO"],
            "config_stub": "meso",
            "switches": {"meso": True, "flow": False, "det": False, "sparse": False, "phy": False},
            "default_module_cfg": {
                "meso": {
                    "cost_model_mode": "explicit_topology_v1",
                }
            },
        },
        {
            "variant_id": "HOPS",
            "internal_experiment_id": "E2",
            "lane_label": "hops_module",
            "mechanism_focus": "hops",
            "public_module_stack": ["ASTRA", "HOPS"],
            "config_stub": "hops",
            "switches": {"meso": False, "flow": True, "det": False, "sparse": False, "phy": False},
            "default_module_cfg": {
                "flow": {
                    "scheduler_mode": "elastic_residency_v3",
                    "reuse_policy": "operand_factored",
                    "admission_policy": "reuse_first",
                    "service_policy": "reuse_first",
                    "exception_lane_policy": "spill",
                }
            },
        },
        {
            "variant_id": "DET",
            "internal_experiment_id": "E3",
            "lane_label": "det_module",
            "mechanism_focus": "det",
            "public_module_stack": ["ASTRA", "DET"],
            "config_stub": "det",
            "switches": {"meso": False, "flow": False, "det": True, "sparse": False, "phy": False},
            "default_module_cfg": {
                "sc_det": {
                    "det_mode": "reorder",
                }
            },
        },
        {
            "variant_id": "SPARSE",
            "internal_experiment_id": "E4",
            "lane_label": "sparse_module",
            "mechanism_focus": "sparse",
            "public_module_stack": ["ASTRA", "SPARSE"],
            "config_stub": "sparse",
            "switches": {"meso": False, "flow": False, "det": False, "sparse": True, "phy": False},
            "default_module_cfg": {
                "sparse": {
                    "use_tau_for_gating": True,
                }
            },
        },
        {
            "variant_id": "PHY",
            "internal_experiment_id": "E5",
            "lane_label": "phy_module",
            "mechanism_focus": "phy",
            "public_module_stack": ["ASTRA", "PHY"],
            "config_stub": "phy",
            "switches": {"meso": False, "flow": False, "det": False, "sparse": False, "phy": True},
            "default_module_cfg": {},
        },
        {
            "variant_id": "FULLER",
            "internal_experiment_id": "E6",
            "lane_label": "fuller_integrated",
            "mechanism_focus": "meso+hops+det+sparse+phy",
            "public_module_stack": ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "PHY"],
            "config_stub": "fuller",
            "switches": {"meso": True, "flow": True, "det": True, "sparse": True, "phy": True},
            "default_module_cfg": {
                "meso": {
                    "cost_model_mode": "explicit_topology_v1",
                },
                "flow": {
                    "scheduler_mode": "elastic_residency_v3",
                    "reuse_policy": "operand_factored",
                    "admission_policy": "reuse_first",
                    "service_policy": "reuse_first",
                    "exception_lane_policy": "spill",
                },
                "sc_det": {
                    "det_mode": "reorder",
                },
                "sparse": {
                    "use_tau_for_gating": True,
                },
            },
        },
    ]


def normalize_variant_descriptor(variant: dict[str, Any]) -> dict[str, Any]:
    explicit_internal = str(
        variant.get("internal_experiment_id") or variant.get("experiment_id") or ""
    ).strip().upper()
    default_descriptor = default_variant_descriptor_for_experiment(explicit_internal)
    explicit_variant_id = str(variant.get("variant_id") or "").strip().upper()
    descriptor = {
        "variant_id": explicit_variant_id or default_descriptor["variant_id"],
        "internal_experiment_id": explicit_internal or default_descriptor["internal_experiment_id"],
        "public_module_stack": copy.deepcopy(
            variant.get("public_module_stack") or default_descriptor["public_module_stack"]
        ),
        "lane_label": str(
            variant.get("lane_label")
            or explicit_variant_id.lower()
            or default_descriptor["variant_id"].lower()
        ).strip(),
        "mechanism_focus": str(
            variant.get("mechanism_focus")
            or explicit_variant_id.lower()
            or default_descriptor["variant_id"].lower()
        ).strip(),
        "config_stub": str(
            variant.get("config_stub")
            or (
                explicit_variant_id.lower()
                if explicit_variant_id
                else default_descriptor["internal_experiment_id"].lower()
            )
        ).strip(),
        "switches": normalize_switches(variant.get("switches")),
        "default_module_cfg": copy.deepcopy(variant.get("default_module_cfg") or {}),
        "accuracy_context_run_id": str(variant.get("accuracy_context_run_id") or "").strip(),
    }
    return descriptor


def variant_lookup_by_internal_id(
    variants: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item["internal_experiment_id"]).upper(): normalize_variant_descriptor(item)
        for item in variants
    }


def variant_lookup_by_public_id(
    variants: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item["variant_id"]).upper(): normalize_variant_descriptor(item)
        for item in variants
    }
