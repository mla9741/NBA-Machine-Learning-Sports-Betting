"""Command-line helper to run the end-to-end NBA betting pipeline."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESS_DATA_DIR = PROJECT_ROOT / "src" / "Process-Data"
TRAIN_MODELS_DIR = PROJECT_ROOT / "src" / "Train-Models"


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_step(title: str, command: Sequence[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    """Run a pipeline step while echoing the command being executed."""
    location = cwd if cwd is not None else PROJECT_ROOT
    print(f"\n>>> {title}")
    print(f"    cwd: {location}")
    print(f"    cmd: {_format_command(command)}")
    if dry_run:
        return

    try:
        subprocess.run(command, cwd=location, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Step '{title}' failed with exit code {exc.returncode}.") from exc


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the data refresh, training, and prediction steps with one command.",
    )
    parser.add_argument(
        "--skip-bdl-games",
        action="store_true",
        help="Skip importing Ball Don't Lie game data.",
    )
    parser.add_argument(
        "--skip-team-data",
        action="store_true",
        help="Skip refreshing team box score data (Get_Data).",
    )
    parser.add_argument(
        "--skip-odds-data",
        action="store_true",
        help="Skip refreshing odds history (Get_Odds_Data).",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip rebuilding the modeling dataset (Create_Games).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip retraining the machine learning models.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip running today's predictions via main.py.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("xgb", "nn"),
        default=("xgb",),
        help="Models to run for the prediction step (defaults to XGBoost only).",
    )
    parser.add_argument(
        "--odds",
        help="Sportsbook to fetch odds for the prediction step (passed to main.py).",
    )
    parser.add_argument(
        "--kelly",
        action="store_true",
        help="Enable Kelly Criterion recommendations when running predictions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    python_exe = sys.executable

    if not args.skip_bdl_games:
        run_step(
            "Import Ball Don't Lie games",
            [python_exe, "-m", "Import_BallDontLie_Games"],
            cwd=PROCESS_DATA_DIR,
            dry_run=args.dry_run,
        )

    if not args.skip_team_data:
        run_step(
            "Refresh team data",
            [python_exe, "-m", "Get_Data"],
            cwd=PROCESS_DATA_DIR,
            dry_run=args.dry_run,
        )

    if not args.skip_odds_data:
        run_step(
            "Refresh odds data",
            [python_exe, "-m", "Get_Odds_Data"],
            cwd=PROCESS_DATA_DIR,
            dry_run=args.dry_run,
        )

    if not args.skip_dataset:
        run_step(
            "Rebuild modeling dataset",
            [python_exe, "-m", "Create_Games"],
            cwd=PROCESS_DATA_DIR,
            dry_run=args.dry_run,
        )

    if not args.skip_training:
        run_step(
            "Train moneyline XGBoost model",
            [python_exe, "-m", "XGBoost_Model_ML"],
            cwd=TRAIN_MODELS_DIR,
            dry_run=args.dry_run,
        )
        run_step(
            "Train totals XGBoost model",
            [python_exe, "-m", "XGBoost_Model_UO"],
            cwd=TRAIN_MODELS_DIR,
            dry_run=args.dry_run,
        )

    if not args.skip_predictions:
        prediction_command = [python_exe, "main.py"]
        models = set(args.models)
        if models == {"xgb", "nn"}:
            prediction_command.append("-A")
        else:
            if "xgb" in models:
                prediction_command.append("-xgb")
            if "nn" in models:
                prediction_command.append("-nn")
        if args.odds:
            prediction_command.append(f"-odds={args.odds}")
        if args.kelly:
            prediction_command.append("-kc")

        run_step(
            "Run daily predictions",
            prediction_command,
            cwd=PROJECT_ROOT,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
