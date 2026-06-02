from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("knitting-sizing-assistant")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _round_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 1:
        return round(value)
    return round(value / multiple) * multiple


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def calculate_K(chest_inches: float, gauge_stitches_per_inch: float) -> str:
    """
    Calculate K, the base stitch count in Elizabeth Zimmermann's Percentage System (EPS).
    K = total stitches around the chest at your gauge.
    """
    k = chest_inches * gauge_stitches_per_inch
    return (
        f"K = {chest_inches} in × {gauge_stitches_per_inch} st/in = {k:.1f} stitches\n\n"
        f"K is the foundation of EPS: it represents the total stitches needed to go "
        f"once around your chest. All other stitch counts are percentages of K."
    )


@mcp.tool()
def adjust_pattern_to_size(
    pattern_chest: float,
    my_chest: float,
    pattern_stitches: int,
    pattern_repeat: int = 1,
) -> str:
    """
    Scale a pattern's stitch count from the pattern's chest size to your chest size.
    Rounds to the nearest pattern_repeat multiple (e.g. 6 for a cable pattern).
    """
    ratio = my_chest / pattern_chest
    raw = pattern_stitches * ratio
    adjusted = _round_to_multiple(raw, pattern_repeat)
    actual_chest = adjusted / (pattern_stitches / pattern_chest)

    lines = [
        f"Pattern chest : {pattern_chest} in  →  Your chest : {my_chest} in",
        f"Scale ratio   : {ratio:.4f}",
        f"Raw stitches  : {pattern_stitches} × {ratio:.4f} = {raw:.1f}",
    ]
    if pattern_repeat > 1:
        lines.append(f"Rounded to nearest {pattern_repeat}-stitch repeat : {adjusted}")
    else:
        lines.append(f"Rounded : {adjusted}")

    lines += [
        f"Actual chest size at adjusted stitch count : {actual_chest:.2f} in",
        "",
        "Tip: the adjusted count keeps your stitch pattern intact while fitting your measurements.",
    ]
    return "\n".join(lines)


@mcp.tool()
def calculate_sweater_EPS(
    chest_inches: float,
    gauge_stitches_per_inch: float,
    gauge_rows_per_inch: float,
    body_length_inches: float,
    arm_length_inches: float,
) -> str:
    """
    Calculate full top-down yoke sweater stitch and row counts using
    Elizabeth Zimmermann's Percentage System (EPS).
    """
    K = chest_inches * gauge_stitches_per_inch

    body_stitches       = round(K)
    sleeve_stitches     = round(K * 0.33)
    neck_stitches       = round(K * 0.40)
    underarm_stitches   = round(K * 0.08)   # each side (4% per side)
    yoke_depth_inches   = chest_inches / 4
    yoke_rows           = round(yoke_depth_inches * gauge_rows_per_inch)
    body_rows           = round(body_length_inches * gauge_rows_per_inch)
    sleeve_rows         = round(arm_length_inches * gauge_rows_per_inch)

    return "\n".join([
        f"{'─' * 48}",
        f"  EPS Sweater Calculator",
        f"  Chest: {chest_inches} in  |  Gauge: {gauge_stitches_per_inch} st/in, {gauge_rows_per_inch} rows/in",
        f"{'─' * 48}",
        f"  K (base stitch count)     : {K:.1f}  →  {body_stitches} st",
        f"",
        f"  YOKE",
        f"    Body cast-on (100% K)   : {body_stitches} st",
        f"    Each sleeve  (33% K)    : {sleeve_stitches} st",
        f"    Neck opening (40% K)    : {neck_stitches} st",
        f"    Underarm each side (4%K): {underarm_stitches} st",
        f"    Yoke depth              : {yoke_depth_inches:.2f} in  →  {yoke_rows} rows",
        f"",
        f"  BODY",
        f"    Body stitches           : {body_stitches} st",
        f"    Body length             : {body_length_inches} in  →  {body_rows} rows",
        f"",
        f"  SLEEVES (each)",
        f"    Sleeve stitches         : {sleeve_stitches} st",
        f"    Arm length              : {arm_length_inches} in  →  {sleeve_rows} rows",
        f"{'─' * 48}",
        f"  All counts are starting points — adjust for your fit preferences.",
    ])


@mcp.tool()
def adjust_for_gauge(
    pattern_gauge_stitches: int,
    pattern_gauge_inches: float,
    my_gauge_stitches: int,
    pattern_total_stitches: int,
    pattern_repeat: int = 1,
) -> str:
    """
    Adjust a pattern's stitch count to compensate for a gauge difference.
    Both gauges are measured over the same number of inches.
    """
    pattern_spi = pattern_gauge_stitches / pattern_gauge_inches
    my_spi      = my_gauge_stitches      / pattern_gauge_inches
    raw         = pattern_total_stitches * (my_spi / pattern_spi)
    adjusted    = _round_to_multiple(raw, pattern_repeat)
    difference  = adjusted - pattern_total_stitches

    lines = [
        f"Pattern gauge : {pattern_gauge_stitches} st = {pattern_gauge_inches} in  "
        f"({pattern_spi:.2f} st/in)",
        f"Your gauge    : {my_gauge_stitches} st = {pattern_gauge_inches} in  "
        f"({my_spi:.2f} st/in)",
        f"Gauge ratio   : {my_spi / pattern_spi:.4f}",
        f"",
        f"Raw adjusted  : {pattern_total_stitches} × {my_spi / pattern_spi:.4f} = {raw:.1f}",
    ]
    if pattern_repeat > 1:
        lines.append(f"Rounded to nearest {pattern_repeat}-stitch repeat : {adjusted}")
    else:
        lines.append(f"Rounded : {adjusted}")

    sign = "+" if difference >= 0 else ""
    lines += [
        f"Change from pattern : {sign}{difference} stitches",
        "",
        (
            "Your gauge is tighter than the pattern — you need more stitches to get the same width."
            if my_spi > pattern_spi else
            "Your gauge is looser than the pattern — you need fewer stitches to get the same width."
            if my_spi < pattern_spi else
            "Your gauge matches the pattern — no adjustment needed."
        ),
    ]
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
