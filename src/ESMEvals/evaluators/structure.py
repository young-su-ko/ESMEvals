from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from .structure_utils import (
    convert_outputs_to_pdb,
    write_pdb,
    extract_mean_plddt,
    sanitize_sequence,
)


@dataclass
class StructurePrediction:
    sequence: str
    pdb_text: str | None
    mean_plddt: float | None
    error: str | None = None


class ESMFoldPredictor:
    """Predicts structures with ESMFold given a sequence."""

    def __init__(
        self,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _esmfold_forward(self, clean_sequence: str) -> tuple[str, dict[str, object]]:
        input_ids = self.tokenizer(
            [clean_sequence],
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        outputs = self.model(input_ids)
        return clean_sequence, outputs

    def predict(
        self,
        sequence: str,
        # output_pdb_path: str | None = None,
    ) -> StructurePrediction:
        clean_sequence = sanitize_sequence(sequence)
        try:
            clean_sequence, outputs = self._esmfold_forward(clean_sequence)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return StructurePrediction(
                sequence=clean_sequence,
                pdb_text=None,
                mean_plddt=None,
                error=f"CUDA out of memory during ESMFold inference due to sequence length: {len(clean_sequence)}.",
            )

        prediction = StructurePrediction(
            sequence=clean_sequence,
            pdb_text=convert_outputs_to_pdb(outputs),
            mean_plddt=extract_mean_plddt(outputs),
        )

        return prediction
