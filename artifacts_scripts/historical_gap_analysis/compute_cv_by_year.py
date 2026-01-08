import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _color_hex(rgb: tuple[float, float, float, float] | tuple[float, float, float]) -> str:
    """Convert a matplotlib RGBA/RGB tuple in [0,1] to an HTML hex string (no '#')."""
    r, g, b = rgb[:3]
    return f"{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def _build_p_color_map(
    p_vals: list[float], *, cmap_name: str = "tab10"
) -> tuple[dict[float, tuple[float, float, float, float]], dict[float, str]]:
    """Return (matplotlib_color, latex_color_name) per p.

    - Colors are assigned deterministically by sorted p.
    - Uses a categorical colormap by default for good separability.
    """
    cmap = plt.get_cmap(cmap_name)
    # tab10 etc are discrete; for continuous cmaps this still works.
    n = max(len(p_vals), 1)
    mpl_colors: dict[float, tuple[float, float, float, float]] = {}
    latex_color_names: dict[float, str] = {}
    for idx, p in enumerate(p_vals):
        rgba = cmap(idx % getattr(cmap, "N", n))
        mpl_colors[p] = rgba
        latex_color_names[p] = f"pcolor{idx}"
    return mpl_colors, latex_color_names


def _safe_cv(mean_val: float, std_val: float) -> float:
    if mean_val is None or std_val is None:
        return np.nan
    try:
        mean_f = float(mean_val)
        std_f = float(std_val)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(mean_f) or not np.isfinite(std_f) or mean_f == 0.0:
        return np.nan
    return std_f / mean_f


def _safe_number(val: float) -> float:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def compute_yearly_cvs_by_p(df: pd.DataFrame, p_values: Iterable[float]) -> pd.DataFrame:
    required_cols = {
        "year",
        "p",
        "denoiser",
        "drop_k",
        "book_gap_size_mean",
        "book_gap_size_std",
        "article_gap_size_mean",
        "article_gap_size_std",
        "landmark_gap_size_mean",
        "landmark_gap_size_std",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["p"] = df["p"].astype(float)
    df["denoiser"] = df["denoiser"].astype(str)
    df["drop_k"] = pd.to_numeric(df["drop_k"], errors="coerce")

    # Restrict to preferred configuration per request
    df_pref = df[(df["denoiser"] == "CCLIP") & (df["drop_k"] == 10)].copy()

    years = sorted(df_pref["year"].unique().tolist())
    p_vals = list(p_values)

    rows: List[dict] = []
    for year in years:
        row: dict = {"year": int(year)}
        for p in p_vals:
            # Find the unique row for this (year, p)
            block = df_pref[(df_pref["year"] == year) & (df_pref["p"] == float(p))]
            if block.empty:
                raise RuntimeError(
                    f"No row found for year={year}, p={p} with denoiser=CCLIP and drop_k=10"
                )
            # If multiple rows exist, take the first (shouldn't happen typically)
            r = block.iloc[0]
            # book → new_books, landmark → prior_books
            row[f"new_books_cv@p={p}"] = _safe_cv(r["book_gap_size_mean"], r["book_gap_size_std"]) 
            row[f"article_cv@p={p}"] = _safe_cv(r["article_gap_size_mean"], r["article_gap_size_std"]) 
            row[f"prior_books_cv@p={p}"] = _safe_cv(r["landmark_gap_size_mean"], r["landmark_gap_size_std"]) 
            row[f"new_books_mean@p={p}"] = _safe_number(r["book_gap_size_mean"])
            row[f"article_mean@p={p}"] = _safe_number(r["article_gap_size_mean"])
            row[f"prior_books_mean@p={p}"] = _safe_number(r["landmark_gap_size_mean"])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def print_results_by_p(yearly: pd.DataFrame, p_values: Iterable[float]) -> None:
    p_vals = list(p_values)
    print("CVs by year:")
    # Per-year rows
    for _, r in yearly.iterrows():
        y = int(r["year"]) if pd.notna(r["year"]) else r["year"]
        parts = []
        for p in p_vals:
            nb = r[f"new_books_cv@p={p}"]
            a = r[f"article_cv@p={p}"]
            pb = r[f"prior_books_cv@p={p}"]
            parts.append(f"p={p}: new_books={nb:.6g} article={a:.6g} prior_books={pb:.6g}")
        print(f"{y}: " + " | ".join(parts))

    print("---")
    print("Gap size means by year:")
    for _, r in yearly.iterrows():
        y = int(r["year"]) if pd.notna(r["year"]) else r["year"]
        parts = []
        for p in p_vals:
            nb = r[f"new_books_mean@p={p}"]
            a = r[f"article_mean@p={p}"]
            pb = r[f"prior_books_mean@p={p}"]
            parts.append(f"p={p}: new_books={nb:.6g} article={a:.6g} prior_books={pb:.6g}")
        print(f"{y}: " + " | ".join(parts))

    print("---")
    # Averages per (p, type)
    avg_parts = []
    for p in p_vals:
        nb_avg = yearly[f"new_books_cv@p={p}"] .mean(skipna=True)
        a_avg = yearly[f"article_cv@p={p}"] .mean(skipna=True)
        pb_avg = yearly[f"prior_books_cv@p={p}"] .mean(skipna=True)
        avg_parts.append(f"p={p}: new_books={nb_avg:.6g} article={a_avg:.6g} prior_books={pb_avg:.6g}")
    print("Average CVs over years: " + " | ".join(avg_parts))

    avg_size_parts = []
    for p in p_vals:
        nb_avg = yearly[f"new_books_mean@p={p}"] .mean(skipna=True)
        a_avg = yearly[f"article_mean@p={p}"] .mean(skipna=True)
        pb_avg = yearly[f"prior_books_mean@p={p}"] .mean(skipna=True)
        avg_size_parts.append(f"p={p}: new_books={nb_avg:.6g} article={a_avg:.6g} prior_books={pb_avg:.6g}")
    print("Average gap sizes over years: " + " | ".join(avg_size_parts))


def _pgf_style_from_matplotlib(ls: str, marker: str) -> str:
    """Map simple matplotlib line/marker styles to PGFPlots styles."""
    style_parts = []
    # Line style
    if ls == "--":
        style_parts.append("dashed")
    elif ls == ":":
        style_parts.append("dotted")
    elif ls == "-.":
        style_parts.append("dashdotted")
    else:
        style_parts.append("solid")

    # Marker style
    marker_map = {
        "o": "mark=*",
        "s": "mark=square*",
        "^": "mark=triangle*",
        "v": "mark=triangle*, mark options={rotate=180}",
        "x": "mark=x",
        "d": "mark=diamond*",
    }
    style_parts.append(marker_map.get(marker, "mark=*"))
    return ", ".join(style_parts)


def _write_latex_plot(
    years: np.ndarray,
    series: list[tuple[str, str, str, np.ndarray]],
    shade_series: list[tuple[str, str, np.ndarray, np.ndarray]] | None,
    p_legend: list[tuple[str, str]] | None,
    type_legend: list[tuple[str, str, str]] | None,
    out_path: Path,
    title: str,
    y_log_scale: bool,
) -> None:
    """Write a minimal PGFPlots (TikZ) plot for the provided series."""
    # Get axis bounds from data
    min_year = int(np.nanmin(years))
    max_year = int(np.nanmax(years))
    
    # Define colors at the top
    color_defs: list[str] = [
        r"\begin{tikzpicture}",
        r"    % Define colors (auto-generated)",
        "",
        r"    % --- MAIN AXIS ---",
    ]
    
    latex_lines: list[str] = color_defs + [
        r"    \begin{axis}[",
        r"        width=\linewidth, height=0.55\linewidth,",
        r"        font=\footnotesize,",
        r"        xlabel={Year},",
        r"        ylabel={CV (std/mean)},",
        r"        grid=major,",
        r"        grid style={dashed,gray!30},",
        r"        xtick distance=5,",
        r"        xticklabel={\pgfmathprintnumber[fixed, precision=0, 1000 sep={}]{\tick}},",
        rf"        xmin={min_year - 1}, xmax={max_year + 1},",
        r"        enlarge x limits=false,",
        r"        legend style={nodes={scale=0.9, transform shape}, at={(0.02,0.98)}, anchor=north west},",
    ]
    if y_log_scale:
        latex_lines.append(r"        ymode=log,")
    latex_lines.append(r"    ]")

    latex_lines.append(rf"        \node at (rel axis cs:0.5,0.98) [anchor=north] {{{title}}};")

    # Add p-legend entries via \addlegendimage (now works with explicit bounds)
    if p_legend:
        for label, color in p_legend:
            latex_lines.append(rf"        \addlegendimage{{color={color}, line width=1.5pt, mark=*, mark size=2pt}}")
            latex_lines.append(rf"        \addlegendentry{{{label}}}")

    # Add shaded bands first so lines draw on top
    if shade_series:
        for label, color, lower, upper in shade_series:
            coords: list[str] = []
            for yr, lo, hi in zip(years, lower, upper):
                if np.isfinite(lo) and np.isfinite(hi):
                    coords.append(f"            ({int(yr)}, {lo:.2f})")
            for yr, lo, hi in zip(reversed(years), reversed(lower), reversed(upper)):
                if np.isfinite(lo) and np.isfinite(hi):
                    coords.append(f"            ({int(yr)}, {hi:.2f})")
            if coords:
                latex_lines.append(rf"        \addplot[draw=none, fill={color}, fill opacity=0.15] coordinates {{")
                latex_lines.extend(coords)
                latex_lines.append(r"        };")

    for label, color, style, values in series:
        latex_lines.append(rf"        \addplot[{style}, color={color}, line width=0.8pt, mark size=1.8pt] coordinates {{")
        for yr, val in zip(years, values):
            if np.isfinite(val):
                latex_lines.append(f"            ({int(yr)}, {val:.2f})")
        latex_lines.append(r"        };")

    latex_lines.append(r"    \end{axis}")

    # Optional secondary legend for types (line/marker styles)
    if type_legend:
        latex_lines.extend([
            "",
            r"    % --- LEGEND 2: LINE STYLES (Overlay) ---",
            r"    \begin{axis}[",
            r"        width=\linewidth, height=0.55\linewidth,",
            r"        hide axis,",
            r"        xmin=0, xmax=1, ymin=0, ymax=1,",
            r"        legend style={nodes={scale=0.9, transform shape}, at={(0.98,0.98)}, anchor=north east}",
            r"    ]",
        ])
        for label, ls, mk in type_legend:
            style = _pgf_style_from_matplotlib(ls, mk)
            latex_lines.append(rf"        \addlegendimage{{{style}, color=black, line width=0.8pt, mark size=1.8pt}}")
            latex_lines.append(rf"        \addlegendentry{{{label}}}")
        latex_lines.append(r"    \end{axis}")

    latex_lines.append(r"\end{tikzpicture}")
    out_path.write_text("\n".join(latex_lines))


def plot_cv_trends(yearly: pd.DataFrame, p_values: Iterable[float], out_path: Path, mode: str = "split", shade_between: bool = True, y_log_scale: bool = False) -> None:
    """Plot CV trends over years, with color encoding for type and line/marker for p. Also emits a PGFPlots .tex."""
    p_vals = list(p_values)
    p_vals = sorted({float(p) for p in p_vals})

    colors_by_p, latex_color_names_by_p = _build_p_color_map(p_vals, cmap_name="tab10")
    styles_by_type = {
        "new_books": ("-", "o"),      # solid + circle
        "prior_books": ("--", "s"),  # dashed + square
    }

    # Shift display years forward by 1 to align with evaluation-year convention used elsewhere
    years = (yearly["year"] + 1).to_numpy()

    # Use 5-year ticks for both Matplotlib and PGF export to avoid dense labels
    min_year = int(np.nanmin(years))
    max_year = int(np.nanmax(years))
    xticks = list(range((min_year // 5) * 5, max_year + 1, 5))

    latex_series: list[tuple[str, str, str, np.ndarray]] = []
    latex_shade: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    latex_type_legend: list[tuple[str, str, str]] | None = None
    latex_p_legend: list[tuple[str, str]] | None = None
    plt.figure(figsize=(12, 6.5))
    
    # Display labels for legend
    type_display = {
        "new_books": "New Books",
        "prior_books": "Prior Books",
    }

    if mode == "combined":
        # Average new_books & prior_books per p: 3 lines
        marker_cycle = ["o", "s", "^", "v", "D", "x", "P", "*", "+", "<", ">"]
        linestyle_cycle = ["-", "--", "-.", ":"]
        for p in p_vals:
            cols = [f"new_books_cv@p={p}", f"prior_books_cv@p={p}"]
            avg_series = yearly[cols].mean(axis=1, skipna=True)
            idx = p_vals.index(p)
            ls = linestyle_cycle[idx % len(linestyle_cycle)]
            mk = marker_cycle[idx % len(marker_cycle)]
            plt.plot(
                years,
                avg_series.to_numpy(),
                linestyle=ls,
                marker=mk,
                color=colors_by_p[p],
                linewidth=1.8,
                markersize=4.5,
                label=f"p={p}",
            )
            latex_series.append(
                (f"p={p}", latex_color_names_by_p[p], _pgf_style_from_matplotlib(ls, mk), avg_series.to_numpy())
            )
        plt.title("Coefficients of Variation of Gap Sizes")
        plt.legend(loc="upper left")

    elif mode == "split":
        # Split new_books vs prior_books: 6 lines (2 types x 3 p's)
        # Color by p, line/marker by type
        latex_type_legend = [
            (type_display["new_books"], styles_by_type["new_books"][0], styles_by_type["new_books"][1]),
            (type_display["prior_books"], styles_by_type["prior_books"][0], styles_by_type["prior_books"][1]),
        ]
        latex_p_legend = [(f"p={p}", latex_color_names_by_p[p]) for p in p_vals]
        if shade_between:
            # Add shaded regions between new_books and prior_books for each p
            for p in p_vals:
                new_books_col = f"new_books_cv@p={p}"
                prior_books_col = f"prior_books_cv@p={p}"
                new_books_series = yearly[new_books_col].to_numpy()
                prior_books_series = yearly[prior_books_col].to_numpy()
                color = colors_by_p[p]
                plt.fill_between(
                    years,
                    new_books_series,
                    prior_books_series,
                    alpha=0.15,
                    color=color,
                    label=f"p={p} (shaded)",
                )
                latex_shade.append((f"p={p} band", latex_color_names_by_p[p], new_books_series, prior_books_series))
        
        for p in p_vals:
            for typ in ["new_books", "prior_books"]:
                col = f"{typ}_cv@p={p}"
                series = yearly[col]
                ls, mk = styles_by_type.get(typ, ("-", "o"))
                plt.plot(
                    years,
                    series.to_numpy(),
                    linestyle=ls,
                    marker=mk,
                    color=colors_by_p[p],
                    linewidth=1.8,
                    markersize=4.5,
                    label=f"{typ}, p={p}",
                )
                latex_series.append(
                    (f"{type_display[typ]}, p={p}", latex_color_names_by_p[p], _pgf_style_from_matplotlib(ls, mk), series.to_numpy())
                )
        plt.title("Coefficients of Variation of Gap Sizes")
        
        # Add two separate legends: one for p (by color), one for type (by style)
        from matplotlib.lines import Line2D
        p_handles = [
            Line2D([0], [0], color=colors_by_p[p], lw=2.0, label=f"p={p}")
            for p in p_vals
        ]
        type_handles = [
            Line2D([0], [0], color="#555555", lw=2.0, linestyle=styles_by_type[t][0], marker=styles_by_type[t][1], label=type_display[t])
            for t in ["new_books", "prior_books"]
        ]
        leg1 = plt.legend(handles=p_handles, loc="upper left")
        leg2 = plt.legend(handles=type_handles, loc="upper right")
        plt.gca().add_artist(leg1)
    else:
        raise ValueError(f"Unknown plot mode: {mode}")

    if y_log_scale:
        plt.yscale("log")
    plt.xticks(xticks)
    plt.xlabel("Year")
    plt.ylabel("CV (std/mean)")
    plt.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # Write LaTeX/PGFPlots version alongside the PNG
    latex_path = out_path.with_suffix(".tex")

    # Prepend auto-generated color definitions for the p palette.
    # We inject these by writing them before calling _write_latex_plot.
    # _write_latex_plot starts with \begin{tikzpicture}, so we provide colors
    # by defining them in a small header and then concatenating.
    color_def_lines: list[str] = []
    for p in p_vals:
        rgba = colors_by_p[p]
        hex_str = _color_hex(rgba)
        color_def_lines.append(rf"    \definecolor{{{latex_color_names_by_p[p]}}}{{HTML}}{{{hex_str}}}")

    _write_latex_plot(
        years=years,
        series=latex_series,
        shade_series=latex_shade if latex_shade else None,
        p_legend=latex_p_legend,
        type_legend=latex_type_legend,
        out_path=latex_path,
        title=("Coefficients of Variation of Gap Sizes"),
        y_log_scale=y_log_scale,
    )

    # Inject the color definitions immediately after \begin{tikzpicture}.
    # This keeps the LaTeX output self-contained for arbitrary p values.
    tex = latex_path.read_text()
    if color_def_lines:
        tex = tex.replace(
            r"\begin{tikzpicture}",
            "\\begin{tikzpicture}\n" + "\n".join(color_def_lines),
            1,
        )
        latex_path.write_text(tex)


def main():
    ap = argparse.ArgumentParser(description="Compute coefficient of variation (std/mean) by year for gap sizes, split by p values, and optionally plot trends.")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/historical_gap_analysis/results/historical_gap_analysis_combined.csv"),
        help="Path to historical_gap_analysis_combined.csv",
    )
    ap.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 2.0],
        help="List of p values to include (default: 0.75 1.0 2.0)",
    )
    ap.add_argument(
        "--plot-out",
        type=Path,
        default=Path("artifacts/historical_gap_analysis/cv_trends.png"),
        help="Output path for the trends plot (PNG)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, do not generate the trends plot",
    )
    ap.add_argument(
        "--plot-mode",
        type=str,
        choices=["combined", "split"],
        default="split",
        help="Plot mode: 'combined' (3 lines by p, averaged over book & landmark) or 'split' (6 lines: book vs landmark by p)",
    )
    ap.add_argument(
        "--min-year",
        type=int,
        default=1999,
        help="Minimum publication year to include (default: 1999)",
    )
    ap.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Maximum publication year to include (default: None for no upper bound)",
    )
    ap.add_argument(
        "--y-log-scale",
        action="store_true",
        help="Plot y-axis on a log scale",
    )
    ap.add_argument(
        "--no-shade-between",
        action="store_true",
        help="Disable shading between new and prior books series for each p (split mode only)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Year range filtering
    if args.min_year is not None:
        df = df[df["year"] >= args.min_year]
    if args.max_year is not None:
        df = df[df["year"] <= args.max_year]

    yearly = compute_yearly_cvs_by_p(df, args.p)
    print_results_by_p(yearly, args.p)
    shade_between = not args.no_shade_between
    if not args.no_plot:
        plot_cv_trends(
            yearly,
            args.p,
            args.plot_out,
            mode=args.plot_mode,
            shade_between=shade_between,
            y_log_scale=args.y_log_scale,
        )
        print(f"Saved plot to: {args.plot_out}")


if __name__ == "__main__":
    main()
