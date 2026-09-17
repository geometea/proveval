"""The 4 required v2 plots, rendered as hand-rolled inline SVG -- no
matplotlib/numpy dependency, consistent with the rest of this repo's
pure-Python statistics (context_analysis_stats.py, controllability_v2_stats.py).
One set of 4 plots is written PER EVALUATOR (never one chart pooling
several evaluators together, matching "never pool different evaluator
models into one inferential sample").

Plots (item 20):
  1. cue_ate_comparison__{evaluator_id}.svg    -- ATE per cue, matched_control
     vs text_only, with 95% CI whiskers.
  2. magnitude_reduction__{evaluator_id}.svg   -- magnitude_reduction and
     residual_text_only_effect per cue.
  3. ambiguity_scatter__{evaluator_id}.svg     -- pair-level D_pair vs
     baseline_strength, with the fitted regression line.
  4. position_effects__{evaluator_id}.svg      -- context effect at each
     display position and the context x position interaction, per cue.
"""

import os

WIDTH, HEIGHT = 760, 420
MARGIN = {"left": 70, "right": 30, "top": 40, "bottom": 90}
COLOR_A = "#3b6fa0"
COLOR_B = "#c96a3a"
COLOR_AXIS = "#333333"
COLOR_GRID = "#dddddd"


def _svg_header(width=WIDTH, height=HEIGHT, title=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="15" font-weight="bold">{_esc(title)}</text>'
    )


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _plot_area():
    x0, x1 = MARGIN["left"], WIDTH - MARGIN["right"]
    y0, y1 = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    return x0, y0, x1, y1


def _scale(value, domain_lo, domain_hi, range_lo, range_hi):
    if domain_hi == domain_lo:
        return (range_lo + range_hi) / 2
    frac = (value - domain_lo) / (domain_hi - domain_lo)
    return range_lo + frac * (range_hi - range_lo)


def grouped_bar_chart_with_ci(title, categories, series, y_label, out_path):
    """categories: list of category labels (x axis groups).
    series: list of {"name": str, "color": str, "values": [v_or_None,...],
    "ci_lo": [.../None,...], "ci_hi": [.../None,...]} -- one entry per
    category, aligned by index with `categories`.
    """
    x0, y0, x1, y1 = _plot_area()
    all_values = [v for s in series for v in s["values"] if v is not None]
    all_values += [v for s in series for v in (s.get("ci_lo") or []) if v is not None]
    all_values += [v for s in series for v in (s.get("ci_hi") or []) if v is not None]
    lo = min(all_values + [0])
    hi = max(all_values + [0])
    pad = (hi - lo) * 0.15 or 0.05
    lo, hi = lo - pad, hi + pad

    svg = [_svg_header(title=title)]
    zero_y = _scale(0, lo, hi, y1, y0)
    svg.append(f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x1}" y2="{zero_y:.1f}" stroke="{COLOR_AXIS}" stroke-width="1"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{COLOR_AXIS}" stroke-width="1"/>')
    svg.append(f'<text x="15" y="{(y0+y1)/2:.0f}" transform="rotate(-90 15 {(y0+y1)/2:.0f})" text-anchor="middle">{_esc(y_label)}</text>')

    n_cat = len(categories)
    n_series = len(series)
    group_width = (x1 - x0) / max(n_cat, 1)
    bar_width = group_width / (n_series + 1)

    for gi, category in enumerate(categories):
        group_x0 = x0 + gi * group_width
        svg.append(f'<text x="{group_x0 + group_width/2:.0f}" y="{y1+15}" text-anchor="middle" transform="rotate(20 {group_x0 + group_width/2:.0f} {y1+15})">{_esc(category)}</text>')
        for si, s in enumerate(series):
            value = s["values"][gi]
            if value is None:
                continue
            bx = group_x0 + (si + 0.5) * bar_width
            by_value = _scale(value, lo, hi, y1, y0)
            top, bottom = (by_value, zero_y) if value >= 0 else (zero_y, by_value)
            svg.append(f'<rect x="{bx - bar_width/2:.1f}" y="{top:.1f}" width="{bar_width*0.9:.1f}" height="{max(bottom-top,0.5):.1f}" fill="{s["color"]}"/>')
            ci_lo = (s.get("ci_lo") or [None]*n_cat)[gi]
            ci_hi = (s.get("ci_hi") or [None]*n_cat)[gi]
            if ci_lo is not None and ci_hi is not None:
                y_ci_lo = _scale(ci_lo, lo, hi, y1, y0)
                y_ci_hi = _scale(ci_hi, lo, hi, y1, y0)
                svg.append(f'<line x1="{bx:.1f}" y1="{y_ci_lo:.1f}" x2="{bx:.1f}" y2="{y_ci_hi:.1f}" stroke="black" stroke-width="1.2"/>')
                svg.append(f'<line x1="{bx-4:.1f}" y1="{y_ci_lo:.1f}" x2="{bx+4:.1f}" y2="{y_ci_lo:.1f}" stroke="black" stroke-width="1.2"/>')
                svg.append(f'<line x1="{bx-4:.1f}" y1="{y_ci_hi:.1f}" x2="{bx+4:.1f}" y2="{y_ci_hi:.1f}" stroke="black" stroke-width="1.2"/>')

    legend_x = x0
    for si, s in enumerate(series):
        ly = y0 - 10
        lx = legend_x + si * 150
        svg.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{s["color"]}"/>')
        svg.append(f'<text x="{lx+16}" y="{ly}" >{_esc(s["name"])}</text>')

    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(svg))


def scatter_with_regression(title, points, intercept, beta, x_label, y_label, out_path):
    """points: list of (x, y) tuples."""
    x0, y0, x1, y1 = _plot_area()
    svg = [_svg_header(title=title)]
    if not points:
        svg.append(f'<text x="{WIDTH/2}" y="{HEIGHT/2}" text-anchor="middle">No data</text></svg>')
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(svg))
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys + [0]), max(ys + [0])
    x_pad = (x_hi - x_lo) * 0.1 or 0.1
    y_pad = (y_hi - y_lo) * 0.15 or 0.05
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad

    svg.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{COLOR_AXIS}"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{COLOR_AXIS}"/>')
    svg.append(f'<text x="{(x0+x1)/2:.0f}" y="{HEIGHT-15}" text-anchor="middle">{_esc(x_label)}</text>')
    svg.append(f'<text x="15" y="{(y0+y1)/2:.0f}" transform="rotate(-90 15 {(y0+y1)/2:.0f})" text-anchor="middle">{_esc(y_label)}</text>')

    for x, y in points:
        cx = _scale(x, x_lo, x_hi, x0, x1)
        cy = _scale(y, y_lo, y_hi, y1, y0)
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{COLOR_A}" fill-opacity="0.7"/>')

    if beta is not None and intercept is not None:
        line_x0, line_x1 = x_lo, x_hi
        line_y0, line_y1 = intercept + beta * line_x0, intercept + beta * line_x1
        cx0, cy0 = _scale(line_x0, x_lo, x_hi, x0, x1), _scale(line_y0, y_lo, y_hi, y1, y0)
        cx1, cy1 = _scale(line_x1, x_lo, x_hi, x0, x1), _scale(line_y1, y_lo, y_hi, y1, y0)
        svg.append(f'<line x1="{cx0:.1f}" y1="{cy0:.1f}" x2="{cx1:.1f}" y2="{cy1:.1f}" stroke="{COLOR_B}" stroke-width="2"/>')

    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(svg))


def _safe_name(evaluator_id):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in evaluator_id)


def write_all_plots(per_evaluator_results, analysis_dir):
    for result in per_evaluator_results:
        eid = _safe_name(result["evaluator_id"])

        # 1. cue ATE: matched_control vs text_only, with CIs
        by_contrast = {}
        for row in result["cue_ates"]:
            by_contrast.setdefault(row["contrast_id"], {})[row["instruction_condition"]] = row
        categories = sorted(by_contrast)
        if categories:
            series = []
            for condition, color in (("matched_control", COLOR_A), ("text_only", COLOR_B)):
                values = [by_contrast[c].get(condition, {}).get("ATE_probability_points") for c in categories]
                ci_lo = [by_contrast[c].get(condition, {}).get("ci_95_lo") for c in categories]
                ci_hi = [by_contrast[c].get(condition, {}).get("ci_95_hi") for c in categories]
                series.append({"name": condition, "color": color, "values": values, "ci_lo": ci_lo, "ci_hi": ci_hi})
            grouped_bar_chart_with_ci(
                f"Cue ATE: matched-control vs text-only ({result['evaluator_id']})",
                categories, series, "ATE (probability points)",
                os.path.join(analysis_dir, f"cue_ate_comparison__{eid}.svg"),
            )

        # 2. magnitude reduction / residual effect by cue
        controllability_by_contrast = {row["contrast_id"]: row for row in result["controllability_effects"]}
        categories2 = sorted(controllability_by_contrast)
        if categories2:
            series2 = [
                {"name": "magnitude_reduction", "color": COLOR_A,
                 "values": [controllability_by_contrast[c].get("magnitude_reduction") for c in categories2]},
                {"name": "residual_text_only_effect", "color": COLOR_B,
                 "values": [controllability_by_contrast[c].get("residual_text_only_effect") for c in categories2]},
            ]
            grouped_bar_chart_with_ci(
                f"Magnitude reduction and residual effect by cue ({result['evaluator_id']})",
                categories2, series2, "Effect (probability points)",
                os.path.join(analysis_dir, f"magnitude_reduction__{eid}.svg"),
            )

        # 3. pair-level context effect vs baseline strength (all cues/conditions pooled visually, not statistically)
        points = [(r["baseline_strength"], r["d_pair"]) for r in result["pair_context_effects"] if r.get("baseline_strength") is not None]
        beta = intercept = None
        ambiguity_rows = result["ambiguity_interactions"]
        if ambiguity_rows:
            # a representative fit for the reference line: the primary contrast's matched_control fit if present, else the first available
            preferred = next((r for r in ambiguity_rows if r["instruction_condition"] == "matched_control"), ambiguity_rows[0])
            intercept, beta = preferred["intercept"], preferred["beta"]
        scatter_with_regression(
            f"Pair-level context effect vs baseline strength ({result['evaluator_id']})",
            points, intercept, beta, "baseline_strength", "D_pair",
            os.path.join(analysis_dir, f"ambiguity_scatter__{eid}.svg"),
        )

        # 4. position-effect diagnostic
        categories4 = sorted({row["contrast_id"] for row in result["position_effects"]})
        if categories4:
            rows_by_contrast = {}
            for row in result["position_effects"]:
                rows_by_contrast.setdefault(row["contrast_id"], []).append(row)
            series4 = [
                {"name": "context_effect_at_story1_as_A", "color": COLOR_A,
                 "values": [_avg_field(rows_by_contrast[c], "context_effect_at_story1_as_a") for c in categories4]},
                {"name": "context_effect_at_story1_as_B", "color": COLOR_B,
                 "values": [_avg_field(rows_by_contrast[c], "context_effect_at_story1_as_b") for c in categories4]},
            ]
            grouped_bar_chart_with_ci(
                f"Position-effect diagnostic ({result['evaluator_id']})",
                categories4, series4, "Context effect (probability points)",
                os.path.join(analysis_dir, f"position_effects__{eid}.svg"),
            )


def _avg_field(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return sum(values) / len(values) if values else None
