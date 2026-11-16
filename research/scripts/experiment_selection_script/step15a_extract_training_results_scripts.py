from pathlib import Path
import shutil
import sys

# ============================================================
# STEP 14A — EXTRACT TRAINING .OUT FILES (PROVENANCE ONLY)
# ============================================================
#
# IMPORTANT:
# The competition primary metric is Log Loss. The training .out files do
# not print Global OOF Log Loss, so they are retained only as execution
# provenance. Competition metrics are recomputed from each experiment's
# root-level oof_predictions.csv in Step 14B.


def main():
    project_root = Path(__file__).resolve().parents[2]
    source_dir = project_root / "data" / "experiments_data"
    output_dir = (
        project_root / "data" / "experiment_selection_data" / "extract_training_results_data"
    )
    expected_count = 60

    print("\n" + "=" * 72)
    print("STEP 14A — EXTRACT TRAINING .OUT FILES (PROVENANCE)")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Source       : {source_dir}")
    print(f"Output       : {output_dir}\n")

    if not source_dir.is_dir():
        print(f"ERROR: experiments directory does not exist:\n{source_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.out"):
        old.unlink()

    out_files = sorted(source_dir.rglob("*.out"))
    print(f"Expected .out files : {expected_count}")
    print(f"Found .out files    : {len(out_files)}\n")

    if len(out_files) != expected_count:
        print(f"ERROR: Expected exactly {expected_count} .out files, found {len(out_files)}.")
        for path in out_files:
            print(path.relative_to(source_dir))
        sys.exit(1)

    for i, source_file in enumerate(out_files, 1):
        rel = source_file.relative_to(source_dir)
        destination_name = "__".join(rel.with_suffix("").parts) + ".out"
        destination = output_dir / destination_name
        shutil.copy2(source_file, destination)
        print(f"[{i:02d}/{expected_count}] {destination_name}")

    final_count = len(list(output_dir.glob("*.out")))
    if final_count != expected_count:
        raise RuntimeError(f"Expected {expected_count} copied .out files, found {final_count}.")

    print("\nSTATUS: PASS")
    print("The 60 .out files were copied for provenance.")
    print("NOTE: Log Loss ranking is NOT derived from these .out files.")


if __name__ == "__main__":
    main()
