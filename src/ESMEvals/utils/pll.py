"""Pseudo-log-likelihood scoring utilities for ESM sequence models."""

import math
import torch


class PseudoLogLikelihoodScorer:
    """Computes token-masked pseudo-log-likelihood for a single protein sequence."""

    def __init__(self, model, alphabet, device: str | torch.device) -> None:
        self.model = model
        self.alphabet = alphabet
        self.device = torch.device(device)
        self.batch_converter = alphabet.get_batch_converter()

    @torch.no_grad()
    def compute_pll(self, sequence: str) -> float:
        if not sequence:
            raise ValueError("Sequence must be non-empty.")

        data = [("protein", sequence)]
        *_, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        log_probs: list[float] = []
        for i in range(len(sequence)):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i + 1] = self.alphabet.mask_idx
            token_probs = torch.log_softmax(
                self.model(batch_tokens_masked)["logits"], dim=-1
            )
            token_log_prob = token_probs[
                0,
                i + 1,
                self.alphabet.get_idx(sequence[i]),
            ].item()
            log_probs.append(token_log_prob)

        return math.fsum(log_probs) / len(sequence)

    @torch.no_grad()
    def compute_approx_pll(
        self,
        sequence: str,
        alpha: float = 0.1,
        beta: float = 0.1,
        epsilon: float = 1e-3,
    ) -> float:
        """Approximate PLL from a single forward pass using alpha-beta smoothing."""
        if not sequence:
            raise ValueError("Sequence must be non-empty.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")
        if beta < 0:
            raise ValueError("beta must be >= 0.")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0.")

        data = [("protein", sequence)]
        *_, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        logits = self.model(batch_tokens)["logits"]  # [1, L, vocab]
        probs = torch.softmax(logits, dim=-1)  # [1, L, vocab]

        scale = (alpha + beta) / alpha
        shift = beta / alpha
        smoothed_probs = torch.clamp(scale * probs - shift, min=epsilon)

        input_ids = batch_tokens[0]  # [L]
        probs_i = smoothed_probs[0]  # [L, vocab]

        cls_idx = self.alphabet.cls_idx
        eos_idx = self.alphabet.eos_idx
        padding_idx = self.alphabet.padding_idx

        valid_mask = (
            (input_ids != cls_idx)
            & (input_ids != eos_idx)
            & (input_ids != padding_idx)
        )

        seq_tokens = input_ids[valid_mask]
        probs_seq = probs_i[valid_mask]

        positions = torch.arange(seq_tokens.size(0), device=self.device)
        selected_probs = probs_seq[positions, seq_tokens]
        log_probs = torch.log(selected_probs)

        return log_probs.sum().item() / log_probs.numel()

    @torch.no_grad()
    def compute_batch_approx_pll(
        self,
        sequences: list[str],
        alpha: float = 0.1,
        beta: float = 0.1,
        epsilon: float = 1e-3,
    ) -> list[float]:
        """Approximate PLL for a batch of sequences from one forward pass."""
        if not sequences:
            raise ValueError("sequences must be non-empty.")
        if any(not sequence for sequence in sequences):
            raise ValueError("All sequences must be non-empty.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")
        if beta < 0:
            raise ValueError("beta must be >= 0.")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0.")

        data = [(f"protein_{i}", sequence) for i, sequence in enumerate(sequences)]
        *_, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        logits = self.model(batch_tokens)["logits"]  # [B, L, vocab]
        probs = torch.softmax(logits, dim=-1)  # [B, L, vocab]

        scale = (alpha + beta) / alpha
        shift = beta / alpha
        smoothed_probs = torch.clamp(scale * probs - shift, min=epsilon)

        cls_idx = self.alphabet.cls_idx
        eos_idx = self.alphabet.eos_idx
        padding_idx = self.alphabet.padding_idx

        scores: list[float] = []
        for i in range(batch_tokens.size(0)):
            input_ids = batch_tokens[i]  # [L]
            probs_i = smoothed_probs[i]  # [L, vocab]

            valid_mask = (
                (input_ids != cls_idx)
                & (input_ids != eos_idx)
                & (input_ids != padding_idx)
            )

            seq_tokens = input_ids[valid_mask]
            probs_seq = probs_i[valid_mask]
            positions = torch.arange(seq_tokens.size(0), device=self.device)
            selected_probs = probs_seq[positions, seq_tokens]
            log_probs = torch.log(selected_probs)
            scores.append(log_probs.sum().item() / log_probs.numel())

        return scores
