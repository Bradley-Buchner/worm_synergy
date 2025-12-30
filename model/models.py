import torch
import torch.nn.functional as F
from torch import nn
import math

# Relational graph transformer encoder layer
class RelationalGTEncoderLayer(nn.Module):
    """
    A Graphormer Encoder Layer modified to use Relational Value Projections.
    This version implements a graph-biased attention mechanism where
    every node attends to every one of its ancestors. The value contribution from each node
    is transformed based on its specific relationship to the querying node (e.g., edge type,
    or in the absence of a direct edge, the most prominent edge type connecting it to the
    querying node).
    """

    def __init__(self, d_model, nhead, value_projections, dim_feedforward=1024, dropout_p=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Store the dictionary of edge-specific value projectors
        self.value_projections = value_projections
        self.num_relation_types = len(self.value_projections)

        # Standard Query, Key, and final Output projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Standard Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Standard Normalization and Dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, src, adj_matrix, attn_bias=None, key_padding_mask=None,
                return_attention=False):
        B, N, D = src.shape

        q = self.q_proj(src).view(B, N, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(src).view(B, N, self.nhead, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_bias is not None:
            attn_scores += attn_bias

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)

        all_v = {
            etype: proj(src).view(B, N, self.nhead, self.d_head).transpose(1, 2)
            for etype, proj in self.value_projections.items()
        }

        attn_output = torch.zeros_like(q)

        for etype_str, v_proj in all_v.items():
            etype_int = int(etype_str)
            relation_mask = (adj_matrix == etype_int).unsqueeze(1)
            masked_attn_weights = attn_weights * relation_mask
            output_for_relation = torch.matmul(masked_attn_weights, v_proj)
            attn_output += output_for_relation

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        attn_output = self.out_proj(attn_output)
        src = self.norm1(src + self.dropout1(attn_output))
        ff_output = self.linear2(self.dropout_ff(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_output))
        if return_attention:
            return src, attn_weights
        return src


# Relational graph transformer for gene pair classification
class SynergyGT(nn.Module):
    """
    Graph transformer architecture modified to use RelationalGTEncoderLayers,
    which learn unique transformations for each edge type.
    """
    def __init__(self,
                 graph_data,
                 params
                 ):
        super().__init__()
        self.num_nodes = graph_data.num_nodes
        self.num_classes = graph_data.pair_effect_type_soft_smoothed.shape[1]
        self.num_edge_types = graph_data.edge_type.max().item()
        self.max_lifespan_dist = graph_data.lifespan_dist.max().item()

        self.d_model = params.get('d_model', 8)
        self.nhead = params.get('num_heads', 4)
        self.num_encoder_layers = params.get('num_layers', 4)
        self.dropout_p = params.get('dropout_p', 0.1)

        self.sum_node_features = params.get('sum_node_features', True)
        self.use_pretrained_gene_embs = params.get('use_pretrained_gene_embs', False)
        self.pretrained_gene_embs_tensor = params.get('pretrained_gene_embs_tensor', None)
        self.fine_tune_gene_emb = params.get('fine_tune_gene_emb', False)
        self.max_spd = params.get('max_spd', 6)
        self.structural_max_dist = params.get('max_spd', 6)
        self.num_degree_bins = params.get('num_degree_bins', 5)
        # self.num_ont_bins = 0

        self.history = {
            "train_loss": [],
            "test_loss": [],
            "test_metrics": [],
        }

        # --- Initialize node embeddings ---
        self.node_identity_embedding = nn.Embedding(self.num_nodes + 1, self.d_model, padding_idx=self.num_nodes)

        if self.pretrained_gene_embs_tensor is not None:
            _, gene_embed_dim = self.pretrained_gene_embs_tensor.shape
            self.gene_embed_dim = gene_embed_dim
            self.pretrained_gene_emb_proj = nn.Linear(gene_embed_dim, self.d_model)
            self.pretrained_node_embedding = nn.Embedding(self.num_nodes + 1, gene_embed_dim, padding_idx=self.num_nodes)
            self.pretrained_node_embedding.weight.data[:self.num_nodes] = self.pretrained_gene_embs_tensor
            self.pretrained_node_embedding.weight.data[self.num_nodes].zero_()
            self.pretrained_node_embedding.weight.requires_grad = self.fine_tune_gene_emb  # set True if you want fine-tuning

        self.dist_uv_embedding = nn.Embedding(self.structural_max_dist + 2, self.d_model)
        self.in_degree_embedding = nn.Embedding(self.num_degree_bins + 1, self.d_model)
        self.out_degree_embedding = nn.Embedding(self.num_degree_bins + 1, self.d_model)
        self.lifespan_dist_embedding = nn.Embedding(self.max_lifespan_dist + 2, self.d_model,
                                                    padding_idx=self.max_lifespan_dist + 1)
        # self.mutual_interactor_emb = nn.Embedding(2, self.d_model)

        # --- Initialize perturbation type embedding ---
        # This embedding is only applied to focal/perturbed nodes.
        # Indices: 0=knockdown, 1=knockout, 2=overexpression, 3=padding_idx (for non-focal nodes)
        self.perturbation_embedding = nn.Embedding(4, self.d_model, padding_idx=3)

        # --- Initialize [PAIR] classification token ---
        self.base_pair_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # --- Initialize embeddings for biasing attention ---
        self.pairwise_dist_embedding = nn.Embedding(self.max_spd + 2, self.nhead, padding_idx=self.max_spd + 1)
        self.edge_type_embedding = nn.Embedding(self.num_edge_types + 1, self.nhead, padding_idx=self.num_edge_types)
        self.mutual_interactor_bias_emb = nn.Embedding(2, self.nhead)
        self.dist_uv_bias_embedding = nn.Embedding(self.structural_max_dist + 2, self.nhead)
        # self.lifespan_bias_emb = nn.Embedding(2, self.nhead)

        # --- Initialize a nn.Linear projection for each unique edge type in the graph ---
        self.value_projections = nn.ModuleDict({
            str(i): nn.Linear(self.d_model, self.d_model) for i in range(self.num_edge_types)
        })
        self.value_projections[str(self.num_edge_types)] = nn.Linear(self.d_model, self.d_model)

        # --- Initialize the transformer encoder's layers ---
        self.encoder_layers = nn.ModuleList([
            RelationalGTEncoderLayer(
                self.d_model, self.nhead, self.value_projections, dim_feedforward=self.d_model * 4, dropout_p=self.dropout_p
            )
            for _ in range(self.num_encoder_layers)
        ])

        # --- Initialize the classifier for node pair classification via the [PAIR] token ---
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model * 2, self.num_classes)
        )

    def forward(self, batch, return_attention=False, return_representation=False):
        B, N = batch['node_ids'].shape

        # --- Get node embeddings for each modality ---
        # Node distance to u/v
        d_u_clamped = torch.clamp(batch['dist_to_u'], max=self.structural_max_dist)
        d_v_clamped = torch.clamp(batch['dist_to_v'], max=self.structural_max_dist)
        structural_embs = (self.dist_uv_embedding(d_u_clamped) + self.dist_uv_embedding(d_v_clamped)) / 2

        # Node degree
        degree_embs = (self.in_degree_embedding(batch['in_degree_binned']) +
                       self.out_degree_embedding(batch['out_degree_binned'])) / 2

        # Node shortest-path distance to a lifespan-associated node (proxy for lifespan association; can be zero)
        clamped_lifespan_dist = torch.clamp(batch['lifespan_dist'], max=self.max_lifespan_dist)
        processed_lifespan_dist = torch.where(
            clamped_lifespan_dist == -1,
            self.lifespan_dist_embedding.padding_idx,
            clamped_lifespan_dist
        )
        lifespan_dist_embs = self.lifespan_dist_embedding(processed_lifespan_dist)

        # lifespan_embs = self.lifespan_association_emb(batch['is_lifespan_gene'])
        # mutual_interactor_embs = self.mutual_interactor_emb(batch['is_mutual_interactor'])
        # gene_embs = self.node_identity_embedding(batch['node_ids'])
        # ont_embs = self.ont_embedding(batch['node_ids'])

        # --- Get node perturbation tag embeddings ---
        # This will be a zero vector for all non-focal nodes.
        node_pert_embs = self.perturbation_embedding(batch['node_perturbations'])

        # --- Get base "from-scratch" node embedding ---
        gene_embs = self.node_identity_embedding(batch['node_ids'])

        # --- Combine all node embeddings via summation ---
        node_features = (gene_embs + structural_embs + degree_embs +
                         node_pert_embs + lifespan_dist_embs)
        # node_features = (gene_embs + degree_embs +
        #                  node_pert_embs + lifespan_dist_embs)

        # Logic for pre-trained gene embeddings option
        if self.pretrained_gene_embs_tensor is not None:
            pretrained_gene_embs = self.pretrained_node_embedding(batch['node_ids'])
            pretrained_gene_embs_proj = self.pretrained_gene_emb_proj(pretrained_gene_embs)
            node_features = node_features + pretrained_gene_embs_proj

        # Get perturbation embeddings for U and V
        u_pert_id = batch['pair_perturbations'][:, 0]
        v_pert_id = batch['pair_perturbations'][:, 1]
        u_pert_emb = self.perturbation_embedding(u_pert_id)
        v_pert_emb = self.perturbation_embedding(v_pert_id)

        # Combine the perturbation embeddings for U and V to the [PAIR] token
        base_pair_token_expanded = self.base_pair_token.expand(B, -1, -1)
        composed_pair_token = (
                base_pair_token_expanded +
                u_pert_emb.unsqueeze(1) +
                v_pert_emb.unsqueeze(1)
        )

        # Aggregate the features of all tokens/nodes (including the [PAIR] token)
        graph_token_features = torch.cat([composed_pair_token, node_features], dim=1)


        # --- Build the attention biases ---
        # 1. Bias by the pairwise distance between nodes
        padded_dist = F.pad(batch['pairwise_dist'], (1, 0, 1, 0), value=self.max_spd + 1)
        pairwise_bias = self.pairwise_dist_embedding(padded_dist)
        pairwise_bias = pairwise_bias.permute(0, 3, 1, 2)

        # # 2. Bias by the average edge type connecting nodes
        # avg_edge_types_padded = F.pad(batch['average_edge_type_encoding'], (1, 0, 1, 0), value=self.num_edge_types)
        # floor_types = torch.floor(avg_edge_types_padded).long()
        # ceil_types = torch.clamp(torch.ceil(avg_edge_types_padded).long(), max=self.num_edge_types)
        # weights = avg_edge_types_padded - floor_types.float()
        # floor_embs = self.edge_type_embedding(floor_types)
        # ceil_embs = self.edge_type_embedding(ceil_types)
        # interpolated_edge_bias = (1.0 - weights.unsqueeze(-1)) * floor_embs + weights.unsqueeze(
        #     -1) * ceil_embs

        # 3. Bias by the lifespan-association status (0/1) of the attention "sender" (key-only bias)
        # cls_pad = torch.zeros(B, 1, dtype=batch['is_lifespan_gene'].dtype,
        #                       device=batch['is_lifespan_gene'].device)
        # padded_lifespan_flags = torch.cat([cls_pad, batch['is_lifespan_gene']], dim=1)
        # lifespan_bias = self.lifespan_bias_emb(padded_lifespan_flags)
        # lifespan_bias = lifespan_bias.permute(0, 2, 1).unsqueeze(2)

        # # 4. Bias by the mutual interactor status (0/1) of the attention "sender" (key-only bias)
        # # Create a (B, 1) tensor of zeros (for the [CLS] token, which is not a mutual interactor)
        # cls_pad = torch.zeros(B, 1, dtype=batch['is_mutual_interactor'].dtype,
        #                       device=batch['is_mutual_interactor'].device)
        # padded_mutual_flags = torch.cat([cls_pad, batch['is_mutual_interactor']], dim=1)
        # mutual_bias = self.mutual_interactor_bias_emb(padded_mutual_flags)
        # mutual_bias = mutual_bias.permute(0, 2, 1).unsqueeze(2)

        # 5. Bias the attention given by [PAIR] based on node proximity to u and v
        # Calculate the bias values for the N nodes
        # (B, N, nhead)
        # d_u_clamped = torch.clamp(batch['dist_to_u'], max=self.structural_max_dist)
        # d_v_clamped = torch.clamp(batch['dist_to_v'], max=self.structural_max_dist)
        cls_node_biases = (self.dist_uv_bias_embedding(d_u_clamped) +
                           self.dist_uv_bias_embedding(d_v_clamped)) / 2

        # We need a full bias matrix (B, nhead, N+1, N+1) initialized to zero
        B, N = batch['node_ids'].shape
        total_tokens = N + 1
        cls_bias_matrix = torch.zeros(B, self.nhead, total_tokens, total_tokens,
                                      device=node_features.device)

        cls_node_biases = cls_node_biases.permute(0, 2, 1)

        # Apply to the first row (the CLS token's attention to all nodes)
        # Index 0 is the [PAIR] token, Indices 1: are the nodes
        cls_bias_matrix[:, :, 0, 1:] = cls_node_biases

        # Add the biases together
        # combined_pairwise_bias = pairwise_dist_bias + interpolated_edge_bias
        # pairwise_bias = combined_pairwise_bias.permute(0, 3, 1, 2)
        # attention_bias = pairwise_bias
        # attention_bias = pairwise_bias + mutual_bias # + lifespan_bias
        attention_bias = pairwise_bias + cls_bias_matrix


        # --- Prepare pairwise relation types for transformer encoder ---
        # Round the average edge type matrix to determine what value projection to use for each nodes pair
        rounded_adj_matrix = torch.round(batch['average_edge_type_encoding']).long()
        adj_matrix_padded = F.pad(rounded_adj_matrix, (1, 0, 1, 0), value=self.num_edge_types)

        # Mask to prevent attention to padding tokens
        token_mask = torch.zeros(B, 1, dtype=torch.bool, device=node_features.device)
        key_padding_mask = torch.cat([token_mask, batch['padding_mask']], dim=1)

        # --- Pass tokens through the transformer encoder ---
        # Collect the current features of all tokens (including the [PAIR] token)
        output = graph_token_features
        all_attn_weights = []  # Create a list to store attention weights
        for layer in self.encoder_layers:
            layer_output = layer(
                output,
                adj_matrix=adj_matrix_padded,
                attn_bias=attention_bias,
                key_padding_mask=key_padding_mask,
                return_attention=return_attention  # option to return attention weights
            )

            if return_attention:
                output, attn_weights = layer_output
                all_attn_weights.append(attn_weights)
            else:
                output = layer_output

        # --- Classify the node pair ---
        # Get the [PAIR] token's representation, which summarizes the global subgraph state
        pair_representation = output[:, 0, :]
        logits = self.classifier(pair_representation)
        outputs = (logits,)

        if return_attention:
            outputs += (all_attn_weights,)

        if return_representation:
            outputs += (pair_representation,)

        # Return a single value if only logits were requested
        return outputs[0] if len(outputs) == 1 else outputs


# Baseline model 1
class UniformBaseline(nn.Module):
    """
    A uniform random distribution baseline model that always predicts a uniform
    label distribution (as logits) regardless of the input features.
    """

    def __init__(self, C):
        super().__init__()
        self.C = C
        # 1. Calculate the uniform probability distribution: (1/C, 1/C, ..., 1/C)
        uniform_dist = torch.full((C,), 1.0 / C)

        # 2. Store the LOG of the uniform distribution (log-probabilities/logits)
        #    to match the output format of NaiveBaseline (and likely your main model).
        self.uniform_log_dist = torch.log(uniform_dist)

    def forward(self, inputs):
        """
        Ignores the inputs and returns the uniform log-distribution repeated
        for the batch size.
        """
        # We need to figure out the batch size from the input tensors.
        # We can just grab any tensor from the input dict to get its size.
        an_input_tensor = next(iter(inputs.values()))
        batch_size = an_input_tensor.size(0)

        # 3. Repeat the uniform log-distribution for each item in the batch.
        #    .to(device) ensures it's on the same device (CPU/GPU) as the input data.
        return self.uniform_log_dist.unsqueeze(0).repeat(batch_size, 1).to(an_input_tensor.device)

# Baseline model 2
class NaiveBaseline(nn.Module):
    """
    A naive baseline model that always predicts the average label
    distribution of the training set.
    """

    def __init__(self, avg_distribution):
        super().__init__()
        # The model's "weights" are just the average distribution.
        # We store the LOG of the distribution because the evaluation function
        # expects logits, and log(probabilities) are a stable form of logits.
        # Adding a small epsilon prevents log(0) = -inf.
        self.avg_log_dist = torch.log(avg_distribution + 1e-9)

    def forward(self, inputs):
        """
        Ignores the inputs and returns the average distribution repeated
        for the batch size.
        """
        # We need to figure out the batch size from the input tensors.
        # We can just grab any tensor from the input dict to get its size.
        an_input_tensor = next(iter(inputs.values()))
        batch_size = an_input_tensor.size(0)

        # Repeat the average distribution for each item in the batch.
        # .to(an_input_tensor.device) ensures it's on the same device (CPU/GPU).
        return self.avg_log_dist.unsqueeze(0).repeat(batch_size, 1).to(an_input_tensor.device)
