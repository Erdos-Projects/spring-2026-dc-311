"""
Central pipeline orchestrator and debug inspector.

PIPELINE COMMANDS
─────────────────
Run the full pipeline end-to-end:
    python run.py                               # default (ward3, lgbm, first_try not needed)
    python run.py --config-name first_try       # use single flat config
    python run.py ward=ward1 model=xgb          # inline overrides

Each stage can also be invoked individually (with its own @hydra.main):
    python -m modeling.data.master  [overrides]
    python -m modeling.search.grid  +search=grid [overrides]
    python -m modeling.search.bayes +search=bayes [overrides]
    python -m modeling.train        [overrides]
    python -m modeling.evaluate     [overrides]

DEBUG / INSPECT COMMANDS
─────────────────────────
    python run.py inspect-master    [overrides]   – summarise the daily series
    python run.py inspect-features  [overrides]   – print feature matrix stats
    python run.py inspect-split     [overrides]   – print split counts by quarter
    python run.py inspect-grid      [--top N]     – print top-N grid results

All inspect commands accept the same Hydra overrides as the pipeline scripts
(e.g. --config-name first_try  ward=ward1  features.d=10).
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ── Hydra config loader (programmatic) ────────────────────────────────────────

def _load_cfg(overrides: list[str]):
    """Load a Hydra DictConfig from a list of override strings."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).parent.absolute() / "configs")

    config_name = "config"
    clean: list[str] = []
    i = 0
    while i < len(overrides):
        if overrides[i] in ("--config-name", "-cn") and i + 1 < len(overrides):
            config_name = overrides[i + 1]
            i += 2
        else:
            clean.append(overrides[i])
            i += 1

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=clean)
    return cfg


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(overrides: list[str]) -> None:
    """
    Execute all pipeline stages in sequence, forwarding Hydra overrides to
    each individual script.
    """
    stages = [
        ("build_daily",   [sys.executable, "-m", "modeling.data.master"]),
        ("grid_search",   [sys.executable, "-m", "modeling.search.grid",  "+search=grid"]),
        ("bayes_search",  [sys.executable, "-m", "modeling.search.bayes", "+search=bayes"]),
        ("train",         [sys.executable, "-m", "modeling.train"]),
        ("evaluate",      [sys.executable, "-m", "modeling.evaluate"]),
    ]

    for name, cmd in stages:
        print(f"\n{'='*60}")
        print(f"  Stage: {name}")
        print(f"{'='*60}")
        full_cmd = cmd + overrides
        result = subprocess.run(full_cmd, check=False)
        if result.returncode != 0:
            print(f"\n[run.py] Stage '{name}' failed (exit code {result.returncode}).")
            sys.exit(result.returncode)

    print("\n[run.py] All stages complete.")


# ── Inspect helpers ────────────────────────────────────────────────────────────

def _inspect_master(overrides: list[str]) -> None:
    cfg = _load_cfg(overrides)
    from modeling.data.master import build_daily

    df = build_daily(cfg)
    print(f"\n=== Daily Series: {cfg.ward.name} ===")
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {list(df.columns)}")
    print(f"Date range : {df['date'].min()} → {df['date'].max()}")
    nulls = df.isnull().sum()
    if nulls.any():
        print(f"Null counts:\n{nulls[nulls > 0]}")
    cols = [c for c in ["pothole_count", "daily_precip", "daily_snow", "daily_ftc"] if c in df.columns]
    print(f"\nDescriptive stats:\n{df[cols].describe().round(3)}")


def _inspect_features(overrides: list[str]) -> None:
    cfg = _load_cfg(overrides)
    from modeling.data.master import build_daily
    from modeling.features import assemble_features

    master_df = build_daily(cfg)
    feat_df = assemble_features(master_df, cfg.features)

    feature_cols = [c for c in feat_df.columns if c not in ("date", "Y")]
    print(f"\n=== Feature Matrix: {cfg.ward.name} ===")
    print(f"Shape        : {feat_df.shape}")
    print(f"Date range   : {feat_df['date'].min()} → {feat_df['date'].max()}")
    print(f"Null count   : {feat_df.isnull().sum().sum()}")
    print(f"\nTarget Y — distribution:\n{feat_df['Y'].describe().round(3)}")
    corrs = feat_df[feature_cols].corrwith(feat_df["Y"]).sort_values(ascending=False)
    print(f"\nTop-10 feature correlations with Y:\n{corrs.head(10).round(3)}")


def _inspect_split(overrides: list[str]) -> None:
    cfg = _load_cfg(overrides)
    import pandas as pd
    from modeling.data.master import build_daily
    from modeling.features import assemble_features
    from modeling.split import make_split

    master_df = build_daily(cfg)
    feat_df = assemble_features(master_df, cfg.features)
    feat_df = make_split(feat_df, cfg.split)
    feat_df["quarter"] = pd.to_datetime(feat_df["date"]).dt.quarter

    print(f"\n=== Split Summary: {cfg.ward.name} ===")
    pivot = feat_df.groupby(["split", "quarter"]).size().unstack(fill_value=0)
    print(pivot)
    print(f"\nTotal counts:\n{feat_df['split'].value_counts()}")


def _inspect_grid(overrides: list[str], top_n: int = 10) -> None:
    cfg = _load_cfg(overrides)
    import pandas as pd

    path = Path("results") / f"grid_{cfg.ward.name}.csv"
    if not path.exists():
        print(f"Grid results not found at {path}.\nRun grid search first.")
        return

    df = pd.read_csv(path)
    print(f"\n=== Top-{top_n} Grid Configurations: {cfg.ward.name} ===")
    print(df.head(top_n).to_string(index=False))


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        default="all",
        choices=["all", "inspect-master", "inspect-features",
                 "inspect-split", "inspect-grid"],
        help="Command to run (default: all — runs the full pipeline)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of rows to show for inspect-grid (default: 10)",
    )

    # parse_known_args so that Hydra overrides (e.g. ward=ward3) pass through
    args, overrides = parser.parse_known_args()

    dispatch = {
        "all":              lambda: run_pipeline(overrides),
        "inspect-master":   lambda: _inspect_master(overrides),
        "inspect-features": lambda: _inspect_features(overrides),
        "inspect-split":    lambda: _inspect_split(overrides),
        "inspect-grid":     lambda: _inspect_grid(overrides, top_n=args.top),
    }
    dispatch[args.cmd]()


if __name__ == "__main__":
    main()
