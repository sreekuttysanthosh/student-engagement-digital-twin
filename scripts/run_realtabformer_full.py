"""Run full RealTabFormer experiments for both missingness strategies."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from realtabformer import REaLTabFormer

from engagement_twin.preprocessing import (
    restore_from_flags,
    restore_random,
)

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
TRAINING_STEPS = 6600
SAMPLE_ROWS = 21328

FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "artifacts" / "realtabformer" / "full"

FLAG_FEATURES = [
    "nevents",
    "ndays_act",
    "nforum_posts",
]


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_generator_input(
    generator_input: pd.DataFrame,
    expected_columns: list[str],
    train_real: pd.DataFrame,
) -> None:
    """Validate the generator input before training."""

    if generator_input.columns.tolist() != expected_columns:
        raise ValueError(
            "Unexpected RealTabFormer input columns.\n"
            f"Expected: {expected_columns}\n"
            f"Received: {generator_input.columns.tolist()}"
        )

    if generator_input.isna().any().any():
        raise ValueError(
            "RealTabFormer training input must be fully imputed."
        )

    if len(generator_input) != len(train_real):
        raise ValueError(
            "Generator input and real training data row counts differ."
        )


def run_experiment(
    strategy: str,
    resume: bool = False,
    recover_checkpoint: bool = False,
) -> None:
    """Train, sample and restore missingness for one strategy."""

    if resume and recover_checkpoint:
        raise ValueError("--resume and --recover-checkpoint cannot be used together.")

    print(f"\nStarting RealTabFormer experiment: {strategy}")

    train_real = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    generator_input_path = (
        DATA_DIR / f"gen_input_{strategy}_target.csv"
    )

    generator_input = pd.read_csv(generator_input_path)

    # Define the expected input schema.
    if strategy == "random":
        expected_columns = FEATURES + [TARGET]

    else:
        expected_columns = (
            FEATURES
            + [f"{col}_missing" for col in FLAG_FEATURES]
            + [TARGET]
        )

    validate_generator_input(
        generator_input,
        expected_columns,
        train_real,
    )

    # Use separate directories and filenames for each strategy.
    model_dir = (
        OUTPUT_DIR / f"full_{strategy}_seed_{SEED}"
    )

    synthetic_path = (
        OUTPUT_DIR
        / f"full_{strategy}_seed_{SEED}_restored.csv"
    )

    report_path = (
        OUTPUT_DIR
        / f"full_{strategy}_seed_{SEED}_report.json"
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)

    # Set reproducibility seeds.
    set_seeds(SEED)

    # Initialize RealTabFormer.
    model = REaLTabFormer(
        model_type="tabular",
        epochs=10,
        batch_size=8,
        random_state=SEED,
        checkpoints_dir=str(model_dir / "checkpoints"),
        samples_save_dir=str(model_dir / "samples"),
        full_save_dir=str(model_dir / "full"),
        max_steps=TRAINING_STEPS,
    )

    print(f"Training rows: {len(generator_input)}")
    print(f"Maximum training steps: {TRAINING_STEPS}")

    recovery_checkpoint = None
    fit_options = {
        "device": "cpu",
        "num_bootstrap": 20,
        "frac": 0.1,
        "frac_max_data": 10_000,
        "qt_max": "compute",
        "gen_rounds": 1,
        "n_critic": 10,
        "gen_kwargs": {},
    }

    if recover_checkpoint:
        recovery_checkpoint = (
            model_dir / "checkpoints" / f"checkpoint-{TRAINING_STEPS}"
        )
        trainer_state_path = recovery_checkpoint / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(
                f"Completed checkpoint state not found: {trainer_state_path}"
            )
        checkpoint_state = json.loads(
            trainer_state_path.read_text(encoding="utf-8")
        )
        if checkpoint_state.get("global_step") != TRAINING_STEPS:
            raise ValueError(
                f"Expected checkpoint step {TRAINING_STEPS}, "
                f"received {checkpoint_state.get('global_step')}"
            )

        # Build the model's preprocessing/schema state, then restore the
        # checkpoint directly instead of continuing sensitivity training.
        fit_options["n_critic"] = 0
        trainer = model.fit(
            generator_input,
            resume_from_checkpoint=str(recovery_checkpoint),
            **fit_options,
        )
        model.model = model.model.from_pretrained(recovery_checkpoint.as_posix())
        if trainer.state.global_step < TRAINING_STEPS:
            raise ValueError(
                f"Recovered trainer stopped at step {trainer.state.global_step}, "
                f"before checkpoint step {TRAINING_STEPS}."
            )
        print(
            f"Recovered exact model weights from checkpoint step "
            f"{TRAINING_STEPS}."
        )
    else:
        model.fit(
            generator_input,
            resume_from_checkpoint=resume,
            **fit_options,
        )

    # Generate synthetic records.
    print(f"Generating {SAMPLE_ROWS} synthetic rows...")

    set_seeds(SEED)

    generated = model.sample(
        n_samples=SAMPLE_ROWS,
        device="cpu",
        gen_batch=128,
    )

    if generated.columns.tolist() != expected_columns:
        raise ValueError(
            "Unexpected generated columns.\n"
            f"Expected: {expected_columns}\n"
            f"Received: {generated.columns.tolist()}"
        )

    if generated[TARGET].isna().any():
        raise ValueError(
            "Generated engagement labels contain missing values."
        )

    if not generated[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Generated engagement labels must be binary (0 or 1)."
        )

    # Restore the original missingness pattern.
    if strategy == "random":

        missing_rates = train_real[FEATURES].isna().mean()

        restored = restore_random(
            generated,
            missing_rates,
            FEATURES,
            seed=SEED,
            target=TARGET,
        )

    else:

        restored = restore_from_flags(
            generated,
            FEATURES,
            FLAG_FEATURES,
            target=TARGET,
        )

    # Validate final synthetic dataset.
    final_columns = FEATURES + [TARGET]

    if restored.columns.tolist() != final_columns:
        raise ValueError(
            f"Unexpected restored columns: {restored.columns.tolist()}"
        )

    if restored[TARGET].isna().any():
        raise ValueError(
            "Restored dataset contains missing target values."
        )

    if not restored[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Restored target contains values other than 0 and 1."
        )

    # Save synthetic data.
    restored.to_csv(synthetic_path, index=False)

    # Save the trained generator.
    model.full_save_dir = model.full_save_dir.as_posix()
    generator_save_dir = model_dir / "realtabformer"
    model.save(generator_save_dir, allow_overwrite=True)
    saved_generator_path = generator_save_dir / model.experiment_id

    # Save experiment metadata.
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "generation_completed",
        "generator": "RealTabFormer",
        "strategy": strategy,
        "seed": SEED,
        "device": "cpu",
        "epochs_requested": 10,
        "max_steps": TRAINING_STEPS,
        "recovered_from_checkpoint": (
            str(recovery_checkpoint.relative_to(ROOT).as_posix())
            if recovery_checkpoint is not None
            else None
        ),
        "training_rows": len(generator_input),
        "generated_rows": len(generated),
        "restored_rows": len(restored),
        "columns": restored.columns.tolist(),
        "missing_values": (
            restored[FEATURES].isna().sum().to_dict()
        ),
        "class_distribution": (
            restored[TARGET]
            .value_counts(normalize=True)
            .sort_index()
            .to_dict()
        ),
        "synthetic_csv": str(synthetic_path.relative_to(ROOT)),
        "generator_model": str(
            saved_generator_path.relative_to(ROOT).as_posix()
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment completed successfully.")
    print(f"Synthetic data: {synthetic_path}")
    print(f"Generator model: {model_dir / 'realtabformer'}")
    print(f"Experiment report: {report_path}")
    print(f"Final dataset shape: {restored.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full RealTabFormer experiments."
    )

    parser.add_argument(
        "--strategy",
        choices=["random", "from_column"],
        required=True,
        help="Missingness restoration strategy to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint for this strategy.",
    )
    parser.add_argument(
        "--recover-checkpoint",
        action="store_true",
        help=(
            "Restore the completed checkpoint without retraining, then save the "
            "generator, synthetic data and generation report."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        args.strategy,
        resume=args.resume,
        recover_checkpoint=args.recover_checkpoint,
    )


if __name__ == "__main__":
    main()