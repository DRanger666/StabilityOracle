# =====================================================================================
# StabilityOracle - Main Model (`model.py`)
#
# Research Context:
# This script defines the `SiameseGraphormer`, the core predictive model of the
# Stability Oracle framework. This architecture is designed to predict the change in
# Gibbs free energy (ΔΔG) upon a single point mutation. Its key innovation lies in
# its "Siamese" design, which processes the wild-type ("from") and mutant ("to")
# states in parallel using shared weights, allowing it to learn a representation
# of the *change* induced by the mutation.
#
# This file integrates the `Backbone` (from `blocks.py`) which processes the
# structural environment, with a novel regression head that interprets the mutation.
# =====================================================================================

from typing import List, Literal
import torch
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from timm.models.vision_transformer import Block as AttentionBlock
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEma

# Research Note: The model imports the `Backbone` and other modules from `blocks.py`.
# The `Backbone` is a graph transformer pre-trained in a self-supervised manner to
# understand the "rules" of protein structure from masked microenvironments.
from StabilityOracle.model import Backbone, Transformer, Mlp


class SiameseGraphormer(nn.Module):
    """
    The main model class for Stability Oracle. This class orchestrates the interaction
    between the pre-trained structural feature extractor (`Backbone`) and a Siamese
    regression head designed to predict ΔΔG.

    Architectural Overview:
    1. A single, pre-trained `Backbone` instance extracts features from the masked
       structural microenvironment (the "socket" left by removing the wild-type residue).
    2. "Structural Amino Acid Embeddings" are derived from the weights of the
       `Backbone`'s final classification layer. These represent the 20 amino acids
       in the context of the learned structural space.
    3. A Siamese attention mechanism (`regression_attn`) processes two parallel inputs:
       - The "from" state: [from_AA_embedding, environment_tokens]
       - The "to" state:   [to_AA_embedding, environment_tokens]
    4. The contextualized output embeddings for the "from" and "to" states are
       subtracted. This delta vector is then passed to a final MLP (`head`) to
       regress the ΔΔG value.
    """
    def __init__(
        self, args, hidden_channel: int = 128, num_class: int = 1, aa_dim: int = 128
    ):
        super().__init__()

        base_model = Backbone

        # Instantiate the pre-trained graph transformer backbone. This module is
        # responsible for encoding the geometry and chemistry of the local atomic
        # environment surrounding the mutation site.
        self.backbone = base_model(
            depth=args.depth,
            _DEFAULT_V_DIM=(aa_dim, 0),
            num_rbf=4,
            embedding_size=aa_dim,
            drop_rate=0,
            atom_drop_rate=0,
            num_dist_mask=1,
            edge_encoding=False,
            use_physical_properties_input=True,
            mask_closest_atom=False,
            dist_thres=8,
            args=args,
        )

        self.aa_dim = aa_dim
        backbone_dim = aa_dim * len(args.return_idx)
        self.max_atom = 513 * 1

        proj0_dim = backbone_dim
        # Research Note on CLS Token Projection: The amino acid embeddings (`aa_dim`) and
        # the backbone output features (`backbone_dim`) may have different dimensions.
        # This linear layer projects the amino acid embeddings to match the backbone's
        # feature dimension, ensuring they can be concatenated for the attention mechanism.
        if not args.clstoken_unify and self.aa_dim == backbone_dim:
            self.proj_clstoken = nn.Identity()
        else:
            self.proj_clstoken = nn.Linear(self.aa_dim, backbone_dim)
        init_values = None
        num_heads = 8

        # Research Note on the Siamese Attention Head: This is the core of the Siamese
        # architecture. It's a set of shared Transformer blocks. By processing both the
        # "from" and "to" sequences with these identical weights, the model learns a
        # consistent function to contextualize an amino acid within its environment.
        # This makes their resulting representations directly comparable and subtractable.
        self.regression_attn = nn.Sequential(
            AttentionBlock(dim=proj0_dim, num_heads=num_heads, init_values=init_values),
            AttentionBlock(dim=proj0_dim, num_heads=num_heads, init_values=init_values),
        )
        self.proj1 = nn.Identity()
        head_dim = proj0_dim

        # The final regression head. It takes the feature delta vector (the result of
        # the subtraction) and maps it to a single scalar value representing ΔΔG.
        self.head = nn.Sequential(
            nn.Linear(head_dim * 1, hidden_channel // 2),
            nn.BatchNorm1d(hidden_channel // 2),
            nn.SiLU(),
            nn.Linear(hidden_channel // 2, num_class),
        )

        self.args = args

        # This option allows for freezing the backbone during fine-tuning. This is a common
        # transfer learning strategy where only the task-specific head is trained,
        # preserving the general-purpose structural features learned during pre-training.
        if args.freeze_backbone:
            self.backbone.requires_grad_(False)

        # A learnable parameter for an additional, potentially "unknown" or "gap" token.
        self.add_aa = nn.Parameter(torch.randn([1, self.aa_dim]) * 0.01)

    def forward(self, feats, atom_types, pp, coords, ca, mask, aa_feats):
        """
        The forward pass implementing the logic described in Figure 1.b of the manuscript.
        """
        # --- 1. Obtain Structural Embeddings ---
        # Research Note: A key design choice. The embeddings for the 20 amino acids are not
        # randomly initialized. Instead, they are the weights of the final linear layer
        # of the pre-trained `Backbone`. This layer was trained to classify masked
        # environments, so its weights inherently encode a structurally-informed
        # representation of each amino acid.
        aa_embeds = self.backbone.dense[-1].weight
        aa_embeds = torch.cat([aa_embeds, self.add_aa], dim=0)

        # --- 2. Encode the Structural Environment ---
        # Pass the masked microenvironment through the backbone to get a set of
        # contextualized feature vectors for each atom in the environment.
        local_env_feats, out = self.backbone(
            feats=feats,
            atom_types=atom_types,
            pp=pp,
            coords=coords,
            ca=ca,
            mask=mask,
            return_idx=self.args.return_idx,
        )

        # --- 3. Prepare Siamese Inputs ---
        # For simplicity in this logic, both 'from' and 'to' start with the same
        # environment features. They will be differentiated by their prepended AA tokens.
        from_feats = local_env_feats
        to_feats = local_env_feats
        bs, num_atom, _ = from_feats.shape

        hidden = from_feats.shape[-1]
        # Project the amino acid embeddings to match the dimension of the backbone's output.
        aa_embeds = self.proj_clstoken(aa_embeds)

        # Select the specific "from" and "to" amino acid embeddings for the current batch
        # based on the input `aa_feats` indices.
        from_aa = aa_embeds[aa_feats[:, 0].long()].reshape(-1, 1, hidden)
        to_aa = aa_embeds[aa_feats[:, 1].long()].reshape(-1, 1, hidden)

        bs = from_feats.shape[0]
        # `mutations` will be 1 in standard inference but handles the 20-way batching
        # from `dataloader.py` during DMS-style prediction.
        mutations = from_aa.shape[0] // bs
        hidden = from_feats.shape[-1]

        # This block handles the 20-way batching for DMS. It expands the single
        # environment representation to match the 20 mutation embeddings.
        from_feats = (
            from_feats.unsqueeze(dim=1)
            .repeat(1, mutations, 1, 1)
            .reshape(mutations * bs, -1, hidden)
        )
        # Construct the final input sequences by prepending the amino acid CLS tokens.
        from_feats = torch.cat([from_aa, from_feats[:, 1:, :]], dim=1)
        to_feats = torch.cat([to_aa, from_feats[:, 1:, :]], dim=1)

        # --- 4. Process with Shared Siamese Attention Head ---
        # Both the "from" and "to" sequences are passed through the *same* `regression_attn` module.
        # This ensures they are processed by an identical function, making their outputs comparable.
        from_feats = self.proj1(self.regression_attn(from_feats).permute(0, 2, 1)).permute(
            0, 2, 1
        )
        to_feats = self.proj1(self.regression_attn(to_feats).permute(0, 2, 1)).permute(0, 2, 1)

        # --- 5. Calculate Delta and Predict ---
        # Research Note: This subtraction is the central operation. Because the "from" and "to"
        # representations were generated by a shared-weight "ruler", their difference
        # vector (`feat_delta`) is a meaningful representation of the mutation's impact.
        # We only care about the change in the CLS token, hence `[:, :1, :]`.
        feat_delta = (to_feats - from_feats)[:, :1, :].mean(dim=1)

        # Pass the delta vector to the final MLP to get the scalar ΔΔG prediction.
        phenotype_delta = self.head(F.dropout(feat_delta, 0.0))

        # The model is trained to predict ΔΔG, but for historical or conventional reasons,
        # the output is scaled by 2. The final evaluation in `pipeline.py` reverses this.
        return 2 * phenotype_delta.reshape(-1), None
