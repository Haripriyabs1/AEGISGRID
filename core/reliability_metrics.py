"""
Reliability Metrics Module
Calculates standard IEEE 1366 Distribution Reliability Indices:
  - SAIFI (System Average Interruption Frequency Index)
  - SAIDI (System Average Interruption Duration Index)
  - CAIDI (Customer Average Interruption Duration Index)
  - ENS (Energy Not Supplied in MWh)
  - ASAI (Average Service Availability Index)
Compares Traditional Manual Crew Restoration vs. Autonomous AI Self-Healing,
and computes Outage Cost Savings based on Value of Lost Load (VoLL).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ReliabilityComparison:
    # Customer counts
    total_customers: int
    interrupted_customers_manual: int
    interrupted_customers_ai: int
    customers_saved_by_ai: int

    # Durations (minutes)
    manual_repair_duration_min: float
    ai_healing_duration_min: float

    # Indices - Manual Restoration
    saifi_manual: float
    saidi_manual_min: float
    caidi_manual_min: float
    ens_manual_mwh: float
    asai_manual_pct: float

    # Indices - AI Self-Healing
    saifi_ai: float
    saidi_ai_min: float
    caidi_ai_min: float
    ens_ai_mwh: float
    asai_ai_pct: float

    # Improvements
    saidi_reduction_percent: float
    ens_saved_mwh: float
    economic_savings_usd: float


def compute_reliability_indices(
    total_load_mw: float,
    deenergized_load_mw: float,
    restored_load_mw: float,
    unserved_load_mw: float,
    manual_repair_time_hours: float = 2.0,  # typical utility dispatch: 120 mins
    ai_restoration_time_seconds: float = 2.5,
    customers_per_mw: int = 800,  # ~800 customers per MW typical
    value_of_lost_load_per_mwh: float = 9500.0  # IEEE standard VoLL ($/MWh)
) -> ReliabilityComparison:
    """
    Computes and compares IEEE 1366 reliability indices for:
      1. Traditional Manual Crew Response (long outage for all downstream loads)
      2. AI-Based Autonomous Self-Healing (near-instant restoration of healthy sections)
    """
    total_cust = max(100, int(total_load_mw * customers_per_mw))
    
    # Downstream customers affected initially
    interrupted_cust_manual = int(deenergized_load_mw * customers_per_mw)
    
    # Under AI self-healing, restored customers only experience a momentary blip (< 1 minute),
    # not a sustained interruption (IEEE defines sustained interruption as > 5 minutes)
    sustained_interrupted_ai = int(unserved_load_mw * customers_per_mw)
    cust_saved = max(0, interrupted_cust_manual - sustained_interrupted_ai)

    manual_time_min = manual_repair_time_hours * 60.0
    ai_time_min = ai_restoration_time_seconds / 60.0

    # 1. Manual Restoration Indices
    # Sustained SAIFI
    saifi_manual = interrupted_cust_manual / total_cust
    # SAIDI = (Customers * Duration) / Total Customers
    customer_minutes_manual = interrupted_cust_manual * manual_time_min
    saidi_manual = customer_minutes_manual / total_cust
    caidi_manual = (saidi_manual / saifi_manual) if saifi_manual > 0 else 0.0

    # Energy Not Supplied (ENS = MW * hours)
    ens_manual = deenergized_load_mw * manual_repair_time_hours

    # ASAI: 8760 hours in a year = 525,600 minutes
    annual_customer_minutes = total_cust * 525600.0
    asai_manual = ((annual_customer_minutes - customer_minutes_manual) / annual_customer_minutes) * 100.0

    # 2. AI Self-Healing Indices
    # Customers in permanently faulted zone still wait for repair; restored customers were healed in seconds
    saifi_ai = sustained_interrupted_ai / total_cust
    
    # Restored customers suffered ai_time_min; permanently unserved suffered manual_time_min
    cust_restored = int(restored_load_mw * customers_per_mw)
    customer_minutes_ai = (sustained_interrupted_ai * manual_time_min) + (cust_restored * ai_time_min)
    saidi_ai = customer_minutes_ai / total_cust
    caidi_ai = (saidi_ai / saifi_ai) if saifi_ai > 0 else 0.0

    ens_ai = (unserved_load_mw * manual_repair_time_hours) + (restored_load_mw * (ai_restoration_time_seconds / 3600.0))
    asai_ai = ((annual_customer_minutes - customer_minutes_ai) / annual_customer_minutes) * 100.0

    # Improvements
    saidi_reduction = ((saidi_manual - saidi_ai) / saidi_manual * 100.0) if saidi_manual > 0 else 0.0
    ens_saved = max(0.0, ens_manual - ens_ai)
    economic_savings = ens_saved * value_of_lost_load_per_mwh

    return ReliabilityComparison(
        total_customers=total_cust,
        interrupted_customers_manual=interrupted_cust_manual,
        interrupted_customers_ai=sustained_interrupted_ai,
        customers_saved_by_ai=cust_saved,
        manual_repair_duration_min=manual_time_min,
        ai_healing_duration_min=ai_time_min,
        saifi_manual=round(saifi_manual, 4),
        saidi_manual_min=round(saidi_manual, 2),
        caidi_manual_min=round(caidi_manual, 2),
        ens_manual_mwh=round(ens_manual, 3),
        asai_manual_pct=round(asai_manual, 5),
        saifi_ai=round(saifi_ai, 4),
        saidi_ai_min=round(saidi_ai, 2),
        caidi_ai_min=round(caidi_ai, 2),
        ens_ai_mwh=round(ens_ai, 3),
        asai_ai_pct=round(asai_ai, 5),
        saidi_reduction_percent=round(saidi_reduction, 1),
        ens_saved_mwh=round(ens_saved, 3),
        economic_savings_usd=round(economic_savings, 2)
    )
