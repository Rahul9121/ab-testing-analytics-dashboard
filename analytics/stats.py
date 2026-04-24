from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)


def compute_experiment_stats(
    df: pd.DataFrame,
    control_label: str,
    treatment_label: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    subset = df[df["variant"].isin([control_label, treatment_label])].copy()
    if subset.empty:
        raise ValueError("No rows found for selected control/treatment labels.")

    stats_df = (
        subset.groupby("variant")["converted"]
        .agg(n="count", conversions="sum", rate="mean")
        .reset_index()
        .sort_values("variant")
    )

    if control_label not in stats_df["variant"].values or treatment_label not in stats_df["variant"].values:
        raise ValueError("Selected labels are not present in the mapped variant column.")

    control_row = stats_df.loc[stats_df["variant"] == control_label].iloc[0]
    treatment_row = stats_df.loc[stats_df["variant"] == treatment_label].iloc[0]

    control_n = int(control_row["n"])
    treatment_n = int(treatment_row["n"])
    control_conversions = int(control_row["conversions"])
    treatment_conversions = int(treatment_row["conversions"])

    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n
    absolute_lift = treatment_rate - control_rate
    relative_lift = absolute_lift / control_rate if control_rate > 0 else np.nan

    _, p_value = proportions_ztest(
        count=[treatment_conversions, control_conversions],
        nobs=[treatment_n, control_n],
        alternative="two-sided",
    )

    z_critical = norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt(
        (control_rate * (1 - control_rate) / control_n)
        + (treatment_rate * (1 - treatment_rate) / treatment_n)
    )
    wald_ci_low = absolute_lift - (z_critical * se_diff)
    wald_ci_high = absolute_lift + (z_critical * se_diff)

    try:
        ci_low, ci_high = confint_proportions_2indep(
            count1=treatment_conversions,
            nobs1=treatment_n,
            count2=control_conversions,
            nobs2=control_n,
            method="wald",
            compare="diff",
            alpha=alpha,
        )
    except Exception:
        ci_low, ci_high = wald_ci_low, wald_ci_high

    stats_df["ci_low"] = stats_df.apply(
        lambda row: proportion_confint(
            count=row["conversions"],
            nobs=row["n"],
            alpha=alpha,
            method="wilson",
        )[0],
        axis=1,
    )
    stats_df["ci_high"] = stats_df.apply(
        lambda row: proportion_confint(
            count=row["conversions"],
            nobs=row["n"],
            alpha=alpha,
            method="wilson",
        )[1],
        axis=1,
    )

    recommended_n: int | None
    try:
        effect_size = abs(proportion_effectsize(control_rate, treatment_rate))
        if effect_size == 0:
            recommended_n = None
        else:
            recommended_n = ceil(
                NormalIndPower().solve_power(effect_size=effect_size, power=0.80, alpha=alpha, ratio=1.0)
            )
    except Exception:
        recommended_n = None

    return {
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "wald_ci_low": float(wald_ci_low),
        "wald_ci_high": float(wald_ci_high),
        "is_significant": bool(p_value < alpha),
        "recommended_n_per_group": recommended_n,
        "group_stats": stats_df,
    }


def build_interpretation(results: dict[str, Any], alpha: float = 0.05) -> str:
    p_value = results["p_value"]
    absolute_lift = results["absolute_lift"]
    relative_lift = results["relative_lift"]
    ci_low = results["ci_low"]
    ci_high = results["ci_high"]
    is_significant = p_value < alpha

    if is_significant and ci_low > 0:
        decision = "✅ Treatment is a statistically significant winner over control."
    elif is_significant and ci_high < 0:
        decision = "⚠️ Treatment is statistically significantly worse than control."
    else:
        decision = "➖ Result is inconclusive at the selected significance threshold."

    recommendation = (
        "Consider running the experiment longer or increasing sample size."
        if not is_significant
        else "You can consider rolling out treatment, with guardrails on business KPIs."
    )

    relative_lift_text = "N/A" if np.isnan(relative_lift) else f"{relative_lift * 100:.2f}%"

    sample_hint = results.get("recommended_n_per_group")
    if sample_hint is None:
        sample_hint_text = "No additional sample-size estimate available."
    else:
        sample_hint_text = f"Estimated sample size for 80% power: ~{sample_hint:,} users per group."

    return (
        f"{decision}\n\n"
        f"- Control conversion rate: **{results['control_rate'] * 100:.2f}%**\n"
        f"- Treatment conversion rate: **{results['treatment_rate'] * 100:.2f}%**\n"
        f"- Absolute lift: **{absolute_lift * 100:.2f}%**\n"
        f"- Relative lift: **{relative_lift_text}**\n"
        f"- p-value: **{p_value:.4g}** (alpha={alpha:.2f})\n"
        f"- Confidence interval for absolute lift: **[{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]**\n\n"
        f"{recommendation}\n\n"
        f"{sample_hint_text}"
    )

