from dataclasses import dataclass

import esm
import torch

from ESMEvals.utils.pll import PseudoLogLikelihoodScorer


@dataclass
class PseudoLogLikelihoodResult:
    sequence: str
    score: float


@dataclass
class FrechetDistanceResult:
    generated_fasta_path: str
    reference_fasta_path: str
    distance: float


class SequenceEvaluator:
    """Sequence-level evaluation methods powered by an ESM language model."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._frechet_calculator = None
        self.pll_scorer = None

    def _load_model(self, model_name: str):
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        model = model.to(self.device)
        model.eval()
        return model, alphabet

    def lazy_load_pll_scorer(self) -> None:
        if self.pll_scorer is not None:
            return

        self.model, self.alphabet = self._load_model(self.model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.pll_scorer = PseudoLogLikelihoodScorer(
            model=self.model,
            alphabet=self.alphabet,
            device=self.device,
        )

    def calculate_pll(self, sequence: str) -> PseudoLogLikelihoodResult:
        """Compute pseudo-log-likelihood for a single sequence."""
        self.lazy_load_pll_scorer()
        score = self.pll_scorer.compute_pll(sequence=sequence)
        return PseudoLogLikelihoodResult(
            sequence=sequence,
            score=score,
        )

    def calculate_approx_pll(
        self,
        sequence: str,
        alpha: float = 0.1,
        beta: float = 0.1,
        epsilon: float = 1e-3,
    ) -> PseudoLogLikelihoodResult:
        """Compute alpha-beta smoothed approximate PLL from one forward pass."""
        self.lazy_load_pll_scorer()
        score = self.pll_scorer.compute_approx_pll(
            sequence=sequence,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
        )
        return PseudoLogLikelihoodResult(
            sequence=sequence,
            score=score,
        )

    def calculate_batch_approx_pll(
        self,
        sequences: list[str],
        alpha: float = 0.1,
        beta: float = 0.1,
        epsilon: float = 1e-3,
    ) -> list[PseudoLogLikelihoodResult]:
        """Compute alpha-beta smoothed approximate PLL for a batch of sequences."""
        self.lazy_load_pll_scorer()
        scores = self.pll_scorer.compute_batch_approx_pll(
            sequences=sequences,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
        )
        return [
            PseudoLogLikelihoodResult(sequence=sequence, score=score)
            for sequence, score in zip(sequences, scores)
        ]

    def calculate_frechet_distance(
        self,
        generated_fasta_path: str,
        reference_fasta_path: str,
    ) -> FrechetDistanceResult:
        """Compute Fréchet distance between generated and reference FASTA files."""
        if self._frechet_calculator is None:
            from plm_fid import FrechetProteinDistance
            self._frechet_calculator = FrechetProteinDistance(device=self.device)

        distance = float(
            self._frechet_calculator.compute_fid(
                generated_fasta_path,
                reference_fasta_path,
            )
        )
        return FrechetDistanceResult(
            generated_fasta_path=generated_fasta_path,
            reference_fasta_path=reference_fasta_path,
            distance=distance,
        )
