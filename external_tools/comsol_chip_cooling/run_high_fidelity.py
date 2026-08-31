"""Generate the compact nonlinear transient set on a qualified local mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nonlinear_cases import high_fidelity_cases
from run import select_comsol
from run_nonlinear import build_dataset, compile_runner
from summarize_high_fidelity import summarize

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = TOOL_ROOT / "data" / "nonlinear_high_fidelity"
LOCAL_PROFILES = ("local-coarse", "local-medium", "local-fine")


def _require_qualified(profile: str, evidence_root: Path) -> None:
    path = evidence_root / "mesh_acceptance.csv"
    if not path.is_file():
        raise ValueError(f"missing mesh qualification evidence: {path}")
    evidence = pd.read_csv(path)
    selected = evidence[evidence["mesh_profile"] == profile]
    if len(selected) != 1 or not bool(selected["all_cases_pass_benchmark"].iloc[0]):
        raise ValueError(
            f"mesh profile {profile} is not benchmark-qualified against its next refinement "
            f"in {path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol-root")
    parser.add_argument("--mesh-profile", choices=LOCAL_PROFILES, default="local-medium")
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-unqualified", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT / "dynamic")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_root = args.evidence_root.resolve()
    if not args.allow_unqualified:
        _require_qualified(args.mesh_profile, evidence_root)
    root, batch, compiler, source_model = select_comsol(args.comsol_root)
    print(f"Using COMSOL: {root}", flush=True)
    compile_runner(compiler)
    summary = build_dataset(
        high_fidelity_cases(),
        batch=batch,
        source_model=source_model,
        data_root=args.data_root.resolve(),
        mesh_profile=args.mesh_profile,
        reuse_raw=args.reuse_raw,
        overwrite=args.overwrite,
    )
    summary_path = args.data_root.resolve() / "qa_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    if args.data_root.resolve() == evidence_root / "dynamic":
        summarize(evidence_root)
        print(f"High-fidelity quality report: {evidence_root / 'quality_report.md'}")
    else:
        print("Skipped derived views because --data-root is outside the evidence root")
    print(f"High-fidelity trajectories: {args.data_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
