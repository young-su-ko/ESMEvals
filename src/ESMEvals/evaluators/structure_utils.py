import torch
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from pathlib import Path


CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(sequence: str) -> str:
    sequence = sequence.upper()
    return "".join([aa if aa in CANONICAL_AMINO_ACIDS else "X" for aa in sequence])

def convert_outputs_to_pdb(outputs: dict[str, object]) -> str:
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdb_lines = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdb_lines.append(to_pdb(pred))
    return "\n".join(pdb_lines)

def write_pdb(pdb_text: str, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pdb_text)

def extract_mean_plddt(outputs: dict[str, object]) -> float:
    plddt_scores = outputs["plddt"][0].detach().to("cpu")
    return float(plddt_scores.mean().item())
