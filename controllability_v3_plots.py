"""v3 plots as hand-rolled inline SVG (no matplotlib), reusing v2's bar-chart
helper. The central figure is the suppression-vs-collateral-distortion
frontier: x = no-context absolute probability shift (drift), y = pooled
magnitude suppression, one point per intervention with 95% CI whiskers on
both axes, Pareto-efficient points filled."""

import os

from controllability_v2_plots import COLOR_A, COLOR_B, COLOR_AXIS, COLOR_GRID, _esc, _scale, _svg_header, grouped_bar_chart_with_ci

WIDTH, HEIGHT = 760, 480
MARGIN = {"left": 80, "right": 30, "top": 40, "bottom": 70}


def frontier_plot(title, points, out_path):
    """points: [{"intervention_id", "x", "x_lo", "x_hi", "y", "y_lo", "y_hi", "pareto_efficient"}]."""
    xs = [v for p in points for v in (p["x"], p["x_lo"], p["x_hi"]) if v is not None]
    ys = [v for p in points for v in (p["y"], p["y_lo"], p["y_hi"]) if v is not None]
    x_lo, x_hi = min([0.0] + xs), max([0.05] + xs)
    y_lo, y_hi = min([0.0] + ys), max([0.05] + ys)
    pad_x, pad_y = (x_hi - x_lo) * 0.08, (y_hi - y_lo) * 0.08
    x_lo, x_hi, y_lo, y_hi = x_lo - pad_x, x_hi + pad_x, y_lo - pad_y, y_hi + pad_y
    px0, py0, px1, py1 = MARGIN["left"], MARGIN["top"], WIDTH - MARGIN["right"], HEIGHT - MARGIN["bottom"]

    def sx(v):
        return _scale(v, x_lo, x_hi, px0, px1)

    def sy(v):
        return _scale(v, y_lo, y_hi, py1, py0)

    parts = [_svg_header(WIDTH, HEIGHT, title)]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = px0 + frac * (px1 - px0)
        gy = py0 + frac * (py1 - py0)
        parts.append(f'<line x1="{gx}" y1="{py0}" x2="{gx}" y2="{py1}" stroke="{COLOR_GRID}"/>')
        parts.append(f'<line x1="{px0}" y1="{gy}" x2="{px1}" y2="{gy}" stroke="{COLOR_GRID}"/>')
        parts.append(f'<text x="{gx}" y="{py1 + 16}" text-anchor="middle" font-size="10">{x_lo + frac * (x_hi - x_lo):.3f}</text>')
        parts.append(f'<text x="{px0 - 6}" y="{py1 - frac * (py1 - py0) + 4}" text-anchor="end" font-size="10">{y_lo + frac * (y_hi - y_lo):.3f}</text>')
    if x_lo <= 0 <= x_hi:
        parts.append(f'<line x1="{sx(0)}" y1="{py0}" x2="{sx(0)}" y2="{py1}" stroke="{COLOR_AXIS}" stroke-dasharray="4,3"/>')
    if y_lo <= 0 <= y_hi:
        parts.append(f'<line x1="{px0}" y1="{sy(0)}" x2="{px1}" y2="{sy(0)}" stroke="{COLOR_AXIS}" stroke-dasharray="4,3"/>')
    parts.append(f'<rect x="{px0}" y="{py0}" width="{px1 - px0}" height="{py1 - py0}" fill="none" stroke="{COLOR_AXIS}"/>')
    parts.append(f'<text x="{(px0 + px1) / 2}" y="{HEIGHT - 20}" text-anchor="middle">No-context judgment distortion (noise-corrected RMS preference shift vs I0)</text>')
    parts.append(f'<text transform="translate(18,{(py0 + py1) / 2}) rotate(-90)" text-anchor="middle">Contextual suppression (|CE_I0| - |CE_I|)</text>')
    for p in points:
        if p["x"] is None or p["y"] is None:
            continue
        cx, cy = sx(p["x"]), sy(p["y"])
        color = COLOR_A if p.get("pareto_efficient") else COLOR_B
        if p["x_lo"] is not None and p["x_hi"] is not None:
            parts.append(f'<line x1="{sx(p["x_lo"])}" y1="{cy}" x2="{sx(p["x_hi"])}" y2="{cy}" stroke="{color}" stroke-width="1.2"/>')
        if p["y_lo"] is not None and p["y_hi"] is not None:
            parts.append(f'<line x1="{cx}" y1="{sy(p["y_lo"])}" x2="{cx}" y2="{sy(p["y_hi"])}" stroke="{color}" stroke-width="1.2"/>')
        fill = color if p.get("pareto_efficient") else "white"
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{cx + 8}" y="{cy - 8}" font-size="12" font-weight="bold">{_esc(p["intervention_id"])}</text>')
    parts.append(f'<text x="{px1}" y="{py0 - 8}" text-anchor="end" font-size="10">filled = Pareto-efficient; whiskers = 95% story-bootstrap CI</text>')
    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def write_all_plots(results, analysis_dir):
    os.makedirs(analysis_dir, exist_ok=True)
    written = []

    frontier = results.get("suppression_distortion_frontier") or []
    if frontier:
        points = [{
            "intervention_id": r["intervention_id"], "x": r["drift_rms_prob_shift_noise_corrected"], "x_lo": r["drift_rms_prob_shift_noise_corrected_ci_95_lo"], "x_hi": r["drift_rms_prob_shift_noise_corrected_ci_95_hi"],
            "y": r["magnitude_suppression"], "y_lo": r["magnitude_suppression_ci_95_lo"], "y_hi": r["magnitude_suppression_ci_95_hi"],
            "pareto_efficient": r["pareto_efficient"],
        } for r in frontier]
        path = os.path.join(analysis_dir, "suppression_distortion_frontier.svg")
        frontier_plot("Suppression vs collateral distortion (pooled over cues)", points, path)
        written.append(path)

    sup = results.get("suppression_by_intervention") or []
    if sup:
        cats = [r["intervention_id"] for r in sup]
        series = [
            {"name": "CE under I0 (matched control)", "color": COLOR_B, "values": [r["CE_control"] for r in sup], "ci_lo": [r["CE_control_ci_95_lo"] for r in sup], "ci_hi": [r["CE_control_ci_95_hi"] for r in sup]},
            {"name": "residual CE under I", "color": COLOR_A, "values": [r["residual_context_effect"] for r in sup], "ci_lo": [r["residual_context_effect_ci_95_lo"] for r in sup], "ci_hi": [r["residual_context_effect_ci_95_hi"] for r in sup]},
        ]
        path = os.path.join(analysis_dir, "context_effect_by_intervention.svg")
        grouped_bar_chart_with_ci("Pooled context effect: matched control vs each intervention", cats, series, "context effect (probability points)", path)
        written.append(path)

    drift = results.get("no_context_drift_by_intervention") or []
    if drift:
        cats = [r["intervention_id"] for r in drift]
        series = [
            {"name": "RMS preference shift (noise-corrected)", "color": COLOR_A, "values": [r["rms_prob_shift_noise_corrected"] for r in drift], "ci_lo": [r["rms_prob_shift_noise_corrected_ci_95_lo"] for r in drift], "ci_hi": [r["rms_prob_shift_noise_corrected_ci_95_hi"] for r in drift]},
            {"name": "excess A/B disagreement", "color": COLOR_B, "values": [r["ab_disagreement_excess"] for r in drift], "ci_lo": [r["ab_disagreement_excess_ci_95_lo"] for r in drift], "ci_hi": [r["ab_disagreement_excess_ci_95_hi"] for r in drift]},
        ]
        path = os.path.join(analysis_dir, "no_context_drift_by_intervention.svg")
        grouped_bar_chart_with_ci("No-context drift vs I0, by intervention", cats, series, "drift", path)
        written.append(path)

    holdout = results.get("held_out_cue_generalization") or []
    if holdout:
        cats = [r["cue_id"] for r in holdout]
        series = [
            {"name": "suppression, full enumeration (I2)", "color": COLOR_A, "values": [r["suppression_full_enumeration"] for r in holdout], "ci_lo": [r["suppression_full_enumeration_ci_95_lo"] for r in holdout], "ci_hi": [r["suppression_full_enumeration_ci_95_hi"] for r in holdout]},
            {"name": "suppression, cue omitted", "color": COLOR_B, "values": [r["suppression_holdout"] for r in holdout], "ci_lo": [r["suppression_holdout_ci_95_lo"] for r in holdout], "ci_hi": [r["suppression_holdout_ci_95_hi"] for r in holdout]},
        ]
        path = os.path.join(analysis_dir, "held_out_cue_generalization.svg")
        grouped_bar_chart_with_ci("Held-out cue generalization (I2 variants)", cats, series, "magnitude suppression", path)
        written.append(path)
    return written
