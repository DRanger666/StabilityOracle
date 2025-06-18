# StabilityOracle - Dataloader
#
# This script is responsible for the crucial first step in the Stability Oracle framework:
# transforming raw protein structure data into a featurized, graph-based representation
# that the Graphormer model can process. The functions herein directly implement the
# "Masked Microenvironment Graph Generation" pipeline described in the manuscript (see Figure 1.a).
# The core idea is to represent the structural context of a mutation, rather than the
# wild-type residue itself.

import json
import torch
import pandas as pd
import numpy as np


# =====================================================================================
# VOCABULARY AND MAPPINGS
# These dictionaries define the discrete vocabulary for atomic and amino acid types,
# allowing them to be converted into integer indices for use in the model's embedding layers.
# =====================================================================================

ELEMENT_MAP = lambda x: {
    "H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "S": 5, "P": 6,
    # Halogens are grouped into a single category as per common practice in structural bioinformatics.
    "F": 7, "Cl": 7, "CL": 7, "Br": 7, "BR": 7, "I": 7,
}.get(x, 8) # Index 8 serves as the "unknown" or "other" category for any non-standard elements.

AA_INDEX = {
   "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLU": 5, "GLN": 6, "GLY": 7,
   "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14, "SER": 15,
   "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
# A utility to convert 3-letter amino acid codes to their standard 1-letter representations.
AA_3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I',
           'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H',
           'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E',
           'TYR': 'Y', 'MET': 'M'}

AMINO_ACIDS = list(AA_INDEX.keys())

def load_embedded_graph(jsonl: dict, device: torch.device=None) -> dict:
    """
    Research Note:
    A utility function for loading data that has already been pre-processed and embedded.
    This bypasses the raw PDB parsing and featurization steps of `load_raw_graph`,
    likely used for faster experimentation cycles, debugging, or inference when the
    featurized graphs have been pre-computed and saved.
    """
    feats = []
    coords = []
    mask = []
    cas = []
    label = []
    from_aas = []
    to_aas = []
    mut_infos = []
    pdb_codes = []


    for data_idx in range(len(jsonl)):
        example = json.loads(jsonl[data_idx])

        mut_infos.append(example["mut_info"])
        pdb_codes.append(example["pdb_id"])
        feats.append(example["input"])
        coords.append(example["coords"])
        mask.append(example["mask"])
        cas.append(example["ca"])
        # The model is trained to predict -ΔΔG, as is conventional.
        label.append(-example["ddg"])
        from_aas.append(example["from"])
        to_aas.append(example["to"])

    return {
        "feats": feats,
        "coords": coords,
        "mask": mask,
        "cas": cas,
        "label": label,
        "from_aas": from_aas,
        "to_aas": to_aas,
        "mut_infos": mut_infos,
        "pdb_codes": pdb_codes,
    }


def load_raw_graph(jsonl: dict, device: torch.device=None) -> dict:
    """
    Research Note:
    This is the primary data processing function that implements the "Masked Microenvironment
    Graph Generation" pipeline (see Figure 1.a in the manuscript). It takes a list of JSONL
    entries, each describing a mutation site in a PDB structure, and engineers the features
    required by the Graphormer model.
    """
    # A fixed size for padding, ensuring all graphs in a batch have the same dimensions.
    max_atoms = 512

    # Initialize lists to hold tensors for each protein in the batch.
    PPs = []        # Physical Properties (Charge, SASA)
    ATs = []        # Atom Types
    COORDs = []     # Atom Coordinates
    masks = []      # Padding Masks
    CAs = []        # Alpha Carbon Coordinates of the mutation site
    labels = []
    mutation_AAs = []
    mut_infos = []
    pdb_codes = []


    for data_idx in range(len(jsonl)):
        data = json.loads(jsonl[data_idx])

        ss = data.get("snapshot", {})
        filename = ss.get("filename", "")

        # The 'atomic_collection' contains the raw atomic data (element, coordinates, etc.).
        atoms = pd.DataFrame.from_dict(data["atomic_collection"])
        ca_idx = data.get("target_alpha_carbon", None)
        # These are the indices of the atoms belonging to the wild-type residue at the mutation site.
        masked_aa_atom_idx = set(data["target"])

        # --- Masked Microenvironment Implementation ---
        # As per the manuscript, we model the structural "socket" by removing the atoms
        # of the wild-type residue. This forces the model to learn the stability contribution
        # from the surrounding environment's interaction with a potential new residue.
        atom_index = {idx for idx in range(len(atoms))}
        atom_index = list(atom_index - masked_aa_atom_idx)

        # Store the coordinates of the alpha-carbon, which serves as the anchor point
        # for the local coordinate system of the microenvironment.
        ca = (
            torch.as_tensor(
                [atoms["x"][ca_idx], atoms["y"][ca_idx], atoms["z"][ca_idx]],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(20, 1) # Repeated for the 20-way mutation batching.
        )

        # Isolate the atoms that constitute the local environment.
        atoms = atoms.iloc[atom_index].reset_index(drop=True)
        # Truncate if the environment is larger than the maximum allowed size.
        if atoms.shape[0] > max_atoms:
            atoms = atoms[: max_atoms]

        # --- Tensor Featurization ---
        # 1. Atomic Coordinates
        coords = torch.as_tensor(atoms[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
        # Pad with zeros to ensure all coordinate tensors have shape (max_atoms, 3).
        coords = (
            torch.cat((coords, torch.zeros([max_atoms - coords.shape[0], 3])), dim=0)
            .float().unsqueeze(0).repeat(20, 1, 1)
        )

        # 2. Atom Types (Discrete Features)
        atom_types = torch.as_tensor(list(map(ELEMENT_MAP, atoms.element)), dtype=torch.long)
        # Pad with zeros.
        atom_types = (
            torch.cat((atom_types, torch.zeros([max_atoms - atom_types.shape[0]]).long()), dim=0)
            .unsqueeze(0).repeat(20, 1)
        )

        # 3. Biophysical Properties (Continuous Features)
        # The paper mentions using additional structural features. Here, partial charge
        # and solvent accessibility (SASA) are used to enrich the node representations.
        add_props = ['_atom_site.fw2_charge', '_atom_site.FreeSASA_value']
        # Data cleaning: handle missing values ('?' or '.') from PDB files.
        for prop in add_props:
            atoms.loc[atoms[prop].isin(['?', '.']), prop] = 0.0
            atoms[prop] = atoms[prop].astype(float)
        pp = torch.as_tensor(atoms[add_props].to_numpy(), dtype=torch.float32)
        # Pad with zeros.
        pp = (
            torch.cat((pp, torch.zeros([max_atoms - pp.shape[0], 2])), dim=0)
            .float().unsqueeze(0).repeat(20, 1, 1)
        )

        # 4. Padding Mask
        # A tensor of 1s and 0s to inform the attention mechanism which nodes are real atoms vs. padding.
        mask = torch.ones([max_atoms]).float().unsqueeze(0).repeat(20, 1)

        # --- 20-Way Mutation Batching for Efficiency ---
        # To avoid running the model 20 times for each site, we create a single batch
        # containing 20 copies of the microenvironment. Each copy is paired with a
        # different target amino acid, allowing for prediction of all 20 ΔΔG values
        # in one forward pass. This is a key contributor to the framework's computational
        # efficiency (see Supplementary Table 1 for performance benchmarks).
        wt_AA = ss['label']
        from_to_pairs = torch.as_tensor(
            [[AA_INDEX[wt_AA], AA_INDEX[to_AA]] for to_AA in AMINO_ACIDS],
            dtype=torch.long,
        )
        pdb_code = filename[:4]
        pos = ss.get('res_seq_num', 0)
        chain_id = ss.get('chain_id', 'A')
        mut_info = [AA_3to1[wt_AA] + str(pos) + AA_3to1[to_AA] + '_' + chain_id for to_AA in AMINO_ACIDS]
        label = ss.get('ddg', np.nan) # Experimental ΔΔG value, if available.

        # Append the processed tensors for this site to the batch lists.
        ATs.append(atom_types)
        COORDs.append(coords)
        CAs.append(ca)
        PPs.append(pp)
        masks.append(mask)
        mutation_AAs.append(from_to_pairs)
        # Append metadata for each of the 20 mutations.
        pdb_codes += [pdb_code] * 20
        mut_infos += mut_info
        labels += [label] * 20


    # Collate the lists of tensors into single large batch tensors.
    return {
        'atom_types': torch.cat(ATs, dim=0),
        'coords': torch.cat(COORDs, dim=0),
        'cas': torch.cat(CAs, dim=0),
        'pp': torch.cat(PPs, dim=0),
        'mask': torch.cat(masks, dim=0),
        'input_aa': torch.cat(mutation_AAs, dim=0),
        'pdb_codes': pdb_codes,
        'mut_infos': mut_infos,
        'label': labels
    }
