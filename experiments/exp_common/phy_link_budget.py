"""Optical link budget utilities for Phase-1 PHY closure."""

from __future__ import annotations

import math
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sum_loss_path_db(loss_path_db: dict[str, Any] | None) -> float:
    """Sum per-component loss entries (dB)."""
    if not loss_path_db:
        return 0.0
    total = 0.0
    for _, val in loss_path_db.items():
        total += _to_float(val, 0.0)
    return total


def compute_pp_crosstalk_db(
    *,
    wdm_channels_n: int,
    xtalk_db: float | None,
    er_db: float | None,
    model: str = "parametric",
    pp_crosstalk_db: float | None = None,
) -> float:
    """Compute crosstalk power penalty (dB).

    Phase-1 default: parametric penalty using a worst-case additive
    interference model:

      PP = 10*log10(1 + (N-1)*X * (1 + 1/ER))

    where:
      X = 10^(xtalk_db/10)
      ER = 10^(er_db/10)  (if provided)
    """
    if pp_crosstalk_db is not None:
        return _to_float(pp_crosstalk_db, 0.0)
    if wdm_channels_n <= 1:
        return 0.0
    xtalk_db = _to_float(xtalk_db, None)
    if xtalk_db is None:
        return 0.0
    xtalk_linear = 10 ** (xtalk_db / 10.0)
    interference = (wdm_channels_n - 1) * xtalk_linear
    if er_db is not None:
        er_linear = 10 ** (_to_float(er_db, 0.0) / 10.0)
        if er_linear > 0:
            interference *= (1.0 + (1.0 / er_linear))
    if model not in {"parametric"}:
        # Unknown model; fall back to parametric.
        pass
    penalty_linear = 1.0 + interference
    if penalty_linear <= 0:
        return 0.0
    return 10.0 * math.log10(penalty_linear)


def compute_link_budget(
    phy_cfg: dict[str, Any],
    *,
    duty_cycle: float = 1.0,
) -> dict[str, float]:
    """Compute link budget and P_laser (dBm + mW).

    Parameters
    ----------
    duty_cycle : float
        SPARSE gating duty-cycle (0..1].  When < 1.0 the effective number
        of simultaneously active WDM channels is reduced, lowering
        PP_crosstalk and thus P_laser.
    """
    wdm_channels_n = int(_to_float(phy_cfg.get("wdm_channels_n"), 1))
    er_db = _to_float(phy_cfg.get("er_db"), 0.0)
    p_sensitivity_dbm = _to_float(phy_cfg.get("p_sensitivity_dbm"), 0.0)
    pp_extinction_db = _to_float(phy_cfg.get("pp_extinction_db"), 0.0)
    margin_db = _to_float(phy_cfg.get("margin_db"), 0.0)
    loss_path_db = sum_loss_path_db(phy_cfg.get("loss_path_db"))

    # Apply SPARSE duty-cycle to get effective active channels.
    dc = max(0.0, min(1.0, float(duty_cycle)))
    effective_wdm = max(1, math.ceil(wdm_channels_n * dc))

    crosstalk_cfg = phy_cfg.get("crosstalk") or {}
    penalty_table_version = (
        crosstalk_cfg.get("phy_penalty_table_version")
        or phy_cfg.get("phy_penalty_table_version")
    )
    pp_crosstalk_db = compute_pp_crosstalk_db(
        wdm_channels_n=effective_wdm,
        xtalk_db=crosstalk_cfg.get("xtalk_db"),
        er_db=er_db,
        model=str(crosstalk_cfg.get("model") or "parametric"),
        pp_crosstalk_db=crosstalk_cfg.get("pp_crosstalk_db"),
    )

    p_laser_dbm = (
        p_sensitivity_dbm
        + loss_path_db
        + pp_crosstalk_db
        + pp_extinction_db
        + margin_db
    )
    p_laser_mw = 10 ** (p_laser_dbm / 10.0)
    return {
        "wdm_channels_n": float(wdm_channels_n),
        "effective_wdm_channels": float(effective_wdm),
        "duty_cycle": float(dc),
        "loss_path_db": float(loss_path_db),
        "pp_crosstalk_db": float(pp_crosstalk_db),
        "p_laser_dbm": float(p_laser_dbm),
        "p_laser_mw": float(p_laser_mw),
        "phy_penalty_table_version": str(penalty_table_version or "parametric-v1"),
    }


__all__ = [
    "compute_link_budget",
    "compute_pp_crosstalk_db",
    "sum_loss_path_db",
]
