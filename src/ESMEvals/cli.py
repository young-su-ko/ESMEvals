import contextlib
import io
import sys
import typer
import warnings
from rich.table import Table
from rich import print

from ESMEvals.evaluators.sequence import SequenceEvaluator
from ESMEvals.evaluators.structure import ESMFoldPredictor

app = typer.Typer(help="CLI for running ESMEvals components.")


def truncate_sequence(sequence: str) -> str:
    return sequence[:10] + "..."


@app.command("pll")
def compute_pll(
    sequence: str = typer.Argument(..., help="Protein amino acid sequence."),
    model_name: str = typer.Option(
        "esm2_t33_650M_UR50D",
        "--model-name",
        help="ESM model name for sequence evaluation.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device to run on, e.g. 'cpu' or 'cuda'. Defaults to auto-detect.",
    ),
    approx: bool = typer.Option(
        False,
        "--approx",
        help="Use approximate PLL (alpha-beta smoothed, single forward pass).",
    ),
    alpha: float = typer.Option(
        0.1,
        "--alpha",
        help="Alpha smoothing parameter (used when --approx is set).",
    ),
    beta: float = typer.Option(
        0.1,
        "--beta",
        help="Beta smoothing parameter (used when --approx is set).",
    ),
    epsilon: float = typer.Option(
        1e-3,
        "--epsilon",
        help="Minimum probability clamp (used when --approx is set).",
    ),
) -> None:
    evaluator = SequenceEvaluator(model_name=model_name, device=device)
    if approx:
        result = evaluator.calculate_approx_pll(
            sequence=sequence,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
        )
    else:
        result = evaluator.calculate_pll(sequence=sequence)

    table = Table()
    table.add_column("Attribute", style="green", no_wrap=True)
    table.add_column("Result", style="white")
    table.add_row("Sequence", truncate_sequence(result.sequence))
    table.add_row("Sequence Length", str(len(result.sequence)))
    table.add_row("PLL", f"{result.score:.3f}")
    print(table)


@app.command("predict-structure")
def predict_structure(
    sequence: str = typer.Argument(..., help="Protein amino acid sequence."),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device to run on, e.g. 'cpu' or 'cuda'. Defaults to auto-detect.",
    ),
    save_path: str | None = typer.Option(
        None,
        "--save-path",
        help="Optional output path. '.pdb' is appended automatically when omitted.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Print model-loading details and additional diagnostic output.",
    ),
) -> None:
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        warnings.filterwarnings(
            "ignore",
            message=".*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*",
            category=UserWarning,
        )

    capture_buffer = io.StringIO()
    if verbose:
        predictor = ESMFoldPredictor(device=device)
        result = predictor.predict(sequence=sequence)
    else:
        with contextlib.redirect_stdout(capture_buffer), contextlib.redirect_stderr(
            capture_buffer
        ):
            predictor = ESMFoldPredictor(device=device)
            result = predictor.predict(sequence=sequence)

    if result.error is not None:
        print(f"Error: {result.error}", style="red")
        raise typer.Exit(code=1)

    saved_status = "Not saved"
    if save_path:
        output_path = save_path if save_path.lower().endswith(".pdb") else f"{save_path}.pdb"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.pdb_text or "")
        saved_status = output_path

    table = Table()
    table.add_column("Attribute", style="blue", no_wrap=True)
    table.add_column("Result", style="white")
    table.add_row("Sequence", truncate_sequence(result.sequence))
    table.add_row("Sequence Length", str(len(result.sequence)))
    table.add_row("Mean PLDDT", f"{result.mean_plddt:.3f}" if result.mean_plddt else "-")
    table.add_row("Output PDB", saved_status)
    print(table)


if __name__ == "__main__":
    app()
