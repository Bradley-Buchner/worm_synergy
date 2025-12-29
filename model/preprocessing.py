import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import k_hop_subgraph, subgraph, degree, to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from collections import deque
from tqdm import tqdm
import math
import os
from torch.nn.utils.rnn import pad_sequence



class PairSubgraphDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def compute_path_sum_matrix(data, output_dir):
    """
    Computes and saves all core attributes required for the graph transformer in one pass.

    This function now calculates:
    1. avg_path_sum_matrix_old.pt: The sum of edge types averaged over all shortest paths.
    """
    print("--- Starting Graph Transformer Attribute Pre-computation ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    N = data.num_nodes
    edge_index = data.edge_index
    edge_type = data.edge_type

    # Compute all-pairs average path sums
    print("\nCalculating All-Pairs Average Shortest Path Sums...")

    adj = [[] for _ in range(N)]
    edge_type_map = {}
    edge_index_cpu = edge_index.cpu()
    edge_type_cpu = edge_type.cpu()

    for i in range(edge_index_cpu.shape[1]):
        u, v = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
        t = edge_type_cpu[i].item()
        adj[u].append(v)
        edge_type_map[(u, v)] = t

    all_pairs_avg_sum_matrix = torch.zeros((N, N), dtype=torch.float32)

    for source_node in tqdm(range(N), desc="Running Augmented BFS for Avg Sums"):

        distances = torch.full((N,), -1, dtype=torch.long)
        path_counts = torch.zeros((N,), dtype=torch.long)
        total_sums = torch.zeros((N,), dtype=torch.long)

        q = deque([source_node])

        distances[source_node] = 0
        path_counts[source_node] = 1

        while q:
            u = q.popleft()

            for v in adj[u]:
                edge_type_val = edge_type_map.get((u, v), 0)

                if distances[v] == -1:
                    distances[v] = distances[u] + 1
                    path_counts[v] = path_counts[u]
                    total_sums[v] = total_sums[u] + (path_counts[u] * edge_type_val)
                    q.append(v)

                elif distances[v] == distances[u] + 1:
                    path_counts[v] += path_counts[u]
                    total_sums[v] += total_sums[u] + (path_counts[u] * edge_type_val)

        average_sums = total_sums.float() / path_counts.float().clamp(min=1)

        all_pairs_avg_sum_matrix[source_node] = average_sums

    torch.save(all_pairs_avg_sum_matrix, os.path.join(output_dir, 'avg_path_sum_matrix.pt'))
    print("All-pairs average path sum matrix saved.")

    print("\n--- All pre-computations finished successfully! ---")


def compute_gt_attributes(data, is_directed, output_dir):
    """
    Computes and saves all core attributes required for the graph transformer in one pass.

    This function now calculates:
    1. in_degree.pt
    2. out_degree.pt
    3. spd_matrix.pt
    4. avg_path_sum_matrix.pt: The sum of edge types averaged over all shortest paths.
    """
    print("--- Starting Graph Transformer Attribute Pre-computation ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    N = data.num_nodes
    edge_index = data.edge_index
    edge_type = data.edge_type

    # 1. Compute and save in-degree and out-degree
    print("Calculating in-degrees and out-degrees...")
    out_degree = degree(edge_index[0], num_nodes=N)
    in_degree = degree(edge_index[1], num_nodes=N)

    torch.save(in_degree, os.path.join(output_dir, 'in_degree.pt'))
    torch.save(out_degree, os.path.join(output_dir, 'out_degree.pt'))
    print("Degrees saved.")

    # 2. Compute SPD matrix using Scipy
    print("\nBuilding graph for shortest path calculation...")
    adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=N)

    print("Running All-Pairs Shortest Path Distances...")
    dist_matrix, _ = shortest_path(
        csgraph=adj_matrix, directed=is_directed, return_predecessors=True
    )
    dist_matrix[np.isinf(dist_matrix)] = -1
    print("SPD computation complete.")

    spd_tensor = torch.from_numpy(dist_matrix).long()

    if is_directed:
        torch.save(spd_tensor, os.path.join(output_dir, 'spd_matrix_directed.pt'))
    else:
        torch.save(spd_tensor, os.path.join(output_dir, 'spd_matrix_undirected.pt'))
    print(f"SPD matrix saved.")

    # 3. Compute all-pairs average path sums
    print("\nCalculating All-Pairs Average Shortest Path Sums...")

    adj = [[] for _ in range(N)]
    edge_type_map = {}
    edge_index_cpu = edge_index.cpu()
    edge_type_cpu = edge_type.cpu()

    for i in range(edge_index_cpu.shape[1]):
        u, v = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
        t = edge_type_cpu[i].item()
        adj[u].append(v)
        edge_type_map[(u, v)] = t

    all_pairs_avg_sum_matrix = torch.zeros((N, N), dtype=torch.float32)

    for source_node in tqdm(range(N), desc="Running Augmented BFS for Avg Sums"):

        distances = torch.full((N,), -1, dtype=torch.long)
        path_counts = torch.zeros((N,), dtype=torch.long)
        total_sums = torch.zeros((N,), dtype=torch.long)

        q = deque([source_node])

        distances[source_node] = 0
        path_counts[source_node] = 1

        while q:
            u = q.popleft()

            for v in adj[u]:
                edge_type_val = edge_type_map.get((u, v), 0)

                if distances[v] == -1:
                    distances[v] = distances[u] + 1
                    path_counts[v] = path_counts[u]
                    total_sums[v] = total_sums[u] + (path_counts[u] * edge_type_val)
                    q.append(v)

                elif distances[v] == distances[u] + 1:
                    path_counts[v] += path_counts[u]
                    total_sums[v] += total_sums[u] + (path_counts[u] * edge_type_val)

        average_sums = total_sums.float() / path_counts.float().clamp(min=1)

        all_pairs_avg_sum_matrix[source_node] = average_sums

    torch.save(all_pairs_avg_sum_matrix, os.path.join(output_dir, 'avg_path_sum_matrix.pt'))
    print("All-pairs average path sum matrix saved.")

    print("\n--- All pre-computations finished successfully! ---")


def bin_logarithmically(degrees, max_val, num_bins=4):
    """
    Converts a tensor of integer values (like degrees) into logarithmic bins.

    This function is designed to handle power-law distributions gracefully by providing
    higher resolution for smaller degree values and grouping larger, rarer degree
    values together.

    - Bin 0 is reserved for degree 0.
    - Bin 1 is reserved for degree 1.
    - The remaining `num_bins - 2` bins are spaced logarithmically from 2 up to `max_val`.
    """
    binned_degrees = torch.zeros_like(degrees, dtype=torch.long)
    binned_degrees[degrees == 1] = 1

    mask = degrees >= 2
    if mask.any():
        num_log_bins = num_bins - 2

        log_min = math.log(2)
        log_max = math.log(max_val)

        boundaries = torch.exp(torch.linspace(log_min, log_max, num_log_bins, device=degrees.device))

        binned_degrees[mask] = torch.bucketize(degrees[mask].float(), boundaries) + 2

    return binned_degrees


def generate_subgraph_samples(data, configs, degree_epsilon=1.0):
    """
    Generates subgraphs comprising the combined one-hop neighborhood of paired genes

    If the initial subgraph has < target_subgraph_size nodes, it augments it.
    - If sampling_mode == 'union':
        Samples from 2-hop intersection, then 2-hop union (failsafe).
    - If sampling_mode == 'intersection':
        Samples from 1-hop union, then 2-hop intersection, then 2-hop union (failsafe).
    All sampling is inversely weighted by degree.
    """

    max_spd = configs.get('max_spd')
    max_dist_uv = configs.get('max_dist_uv')
    k_hop = configs.get('k_hop')
    sampling_mode = configs.get('sampling_mode')
    target_subgraph_size = configs.get('target_subgraph_size')
    max_in_degree = configs.get('max_in_degree')
    max_out_degree = configs.get('max_out_degree')
    num_degree_bins = configs.get('num_degree_bins')
    attributes_dir = configs.get('attributes_dir')
    output_path = configs.get('output_path')
    device = configs.get('device')

    print(f"--- Starting Subgraph Pre-processing ---")
    print(f"Mode: {sampling_mode}, Target Augmentation Size: {target_subgraph_size}, Device: {device}")
    data = data.to(device)

    # 1. Load attributes
    print(f"Loading pre-computed attributes from {attributes_dir}...")
    try:
        spd_tensor = torch.load(os.path.join(attributes_dir, 'spd_matrix.pt'), weights_only=False, map_location=device)
        path_sum_tensor = torch.load(os.path.join(attributes_dir, 'avg_path_sum_matrix.pt'), weights_only=False,
                                     map_location=device)
        in_degree_tensor = torch.load(os.path.join(attributes_dir, 'in_degree.pt'), weights_only=False,
                                      map_location=device)
        out_degree_tensor = torch.load(os.path.join(attributes_dir, 'out_degree.pt'), weights_only=False,
                                       map_location=device)
        total_degree_tensor = in_degree_tensor + out_degree_tensor
    except FileNotFoundError as e:
        print(f"Error: Attribute file not found: {e}. Check attributes_dir.")
        raise
    except Exception as e:
        print(f"Error loading attributes: {e}")
        raise

    processed_samples = []
    num_pairs = data.pair_pert_group_index.shape[1]

    default_dist = max_spd + 1

    print(f"\nProcessing {num_pairs} pairs...")
    max_edge_type_val = 0
    if hasattr(data, 'edge_type') and data.edge_type is not None and data.edge_type.numel() > 0:
        max_edge_type_val = data.edge_type.max().item()
    else:
        print("Warning: data.edge_type missing/empty. Assuming num_edge_types = 1.")
    num_edge_types = max_edge_type_val + 1

    has_soft_targets = hasattr(data, 'pair_effect_type_soft') and data.pair_effect_type_soft is not None
    has_smoothed_targets = hasattr(data,
                                   'pair_effect_type_soft_smoothed') and data.pair_effect_type_soft_smoothed is not None
    # if has_soft_targets or has_smoothed_targets:
    #     print("Target soft/smoothed data found.")
    # else:
    #     print("Target soft/smoothed data NOT found.")

    # Main looop
    for i in tqdm(range(num_pairs), desc="Building Subgraphs"):

        u, pert_u, v, pert_v = data.pair_pert_group_index[:, i].tolist()
        pair_tensor = torch.tensor([u, v], device=device)
        pair_pert_tensor = torch.tensor([pert_u, pert_v], device=device)

        # Calculate mutual 1-hop interactors
        nodes_1_hop_u, _, _, _ = k_hop_subgraph(u, 1, data.edge_index, False)
        nodes_1_hop_v, _, _, _ = k_hop_subgraph(v, 1, data.edge_index, False)

        # Find intersection (excluding u and v themselves potentially, though k_hop includes center node)
        cat_1_hop = torch.cat([nodes_1_hop_u, nodes_1_hop_v])
        uniq_1_hop, counts_1_hop = torch.unique(cat_1_hop, return_counts=True)
        mutual_1_hop_nodes = uniq_1_hop[counts_1_hop > 1]
        # Remove u and v if they are in the mutual list (they trivially intersect)
        mutual_1_hop_nodes = mutual_1_hop_nodes[(mutual_1_hop_nodes != u) & (mutual_1_hop_nodes != v)]

        # Initial subgraph node selection
        if sampling_mode == 'union':
            initial_subgraph_nodes, _, _, _ = k_hop_subgraph(
                node_idx=pair_tensor, num_hops=k_hop, edge_index=data.edge_index, relabel_nodes=False
            )
        elif sampling_mode == 'intersection':
            nodes_k_hop_u, _, _, _ = k_hop_subgraph(u, k_hop, data.edge_index, False)
            nodes_k_hop_v, _, _, _ = k_hop_subgraph(v, k_hop, data.edge_index, False)
            cat_k_hop = torch.cat([nodes_k_hop_u, nodes_k_hop_v])
            uniq_k_hop, counts_k_hop = torch.unique(cat_k_hop, return_counts=True)
            intersect_nodes = uniq_k_hop[counts_k_hop > 1]
            initial_subgraph_nodes = torch.unique(torch.cat([intersect_nodes, pair_tensor]))
        else:
            raise ValueError(f"Unknown sampling_mode: '{sampling_mode}'.")

        N_initial = initial_subgraph_nodes.size(0)
        subgraph_nodes = initial_subgraph_nodes

        # Augmentation section
        if N_initial < target_subgraph_size:

            if sampling_mode == 'intersection':
                # Priority: 1-hop union -> 2-hop intersect -> 2-hop union (failsafe)

                current_N = N_initial

                # Augmentation step 1 (intersection mode): 1-hop union ---
                num_to_sample_1_union = target_subgraph_size - current_N

                # Use 1-hop nodes from mutual interactor calculation
                nodes_1_hop_union = torch.unique(cat_1_hop)

                is_already_present_mask_1_union = torch.isin(nodes_1_hop_union, subgraph_nodes)
                candidate_nodes_1_union = nodes_1_hop_union[~is_already_present_mask_1_union]
                num_candidates_1_union = candidate_nodes_1_union.size(0)

                sampled_1_union_nodes = torch.tensor([], dtype=initial_subgraph_nodes.dtype, device=device)

                if num_candidates_1_union > 0:
                    candidate_degrees_1_union = total_degree_tensor[candidate_nodes_1_union].float()
                    candidate_weights_1_union = 1.0 / (candidate_degrees_1_union + degree_epsilon)
                    candidate_weights_1_union.clamp_min_(1e-6)

                    actual_num_to_sample_1_union = min(num_to_sample_1_union, num_candidates_1_union)

                    if candidate_weights_1_union.sum() <= 1e-6:
                        perm = torch.randperm(num_candidates_1_union, device=device)
                        sampled_indices = perm[:actual_num_to_sample_1_union]
                    else:
                        try:
                            sampled_indices = torch.multinomial(candidate_weights_1_union, actual_num_to_sample_1_union,
                                                                replacement=False)
                        except RuntimeError:
                            perm = torch.randperm(num_candidates_1_union, device=device)
                            sampled_indices = perm[:actual_num_to_sample_1_union]

                    sampled_1_union_nodes = candidate_nodes_1_union[sampled_indices]

                subgraph_nodes = torch.unique(torch.cat([subgraph_nodes, sampled_1_union_nodes]))
                current_N = subgraph_nodes.size(0)

                # Augmentation step 2 (intersection mode): 2-hop intersection ---
                cat_2_hop = None  # Initialize in case this step is skipped
                if current_N < target_subgraph_size:
                    num_to_sample_2_intersect = target_subgraph_size - current_N

                    nodes_2_hop_u, _, _, _ = k_hop_subgraph(u, 2, data.edge_index, False)
                    nodes_2_hop_v, _, _, _ = k_hop_subgraph(v, 2, data.edge_index, False)
                    cat_2_hop = torch.cat([nodes_2_hop_u, nodes_2_hop_v])

                    uniq_2_hop, counts_2_hop = torch.unique(cat_2_hop, return_counts=True)
                    nodes_2_hop_intersect = uniq_2_hop[counts_2_hop > 1]

                    is_already_present_mask_2_intersect = torch.isin(nodes_2_hop_intersect,
                                                                     subgraph_nodes)
                    candidate_nodes_2_intersect = nodes_2_hop_intersect[~is_already_present_mask_2_intersect]
                    num_candidates_2_intersect = candidate_nodes_2_intersect.size(0)

                    sampled_2_intersect_nodes = torch.tensor([], dtype=initial_subgraph_nodes.dtype, device=device)

                    if num_candidates_2_intersect > 0:
                        candidate_degrees_intersect = total_degree_tensor[candidate_nodes_2_intersect].float()
                        candidate_weights_intersect = 1.0 / (candidate_degrees_intersect + degree_epsilon)
                        candidate_weights_intersect.clamp_min_(1e-6)

                        actual_num_to_sample_intersect = min(num_to_sample_2_intersect, num_candidates_2_intersect)

                        if candidate_weights_intersect.sum() <= 1e-6:
                            perm = torch.randperm(num_candidates_2_intersect, device=device)
                            sampled_indices = perm[:actual_num_to_sample_intersect]
                        else:
                            try:
                                sampled_indices = torch.multinomial(candidate_weights_intersect,
                                                                    actual_num_to_sample_intersect, replacement=False)
                            except RuntimeError:
                                perm = torch.randperm(num_candidates_2_intersect, device=device)
                                sampled_indices = perm[:actual_num_to_sample_intersect]

                        sampled_2_intersect_nodes = candidate_nodes_2_intersect[sampled_indices]

                    subgraph_nodes = torch.unique(torch.cat([subgraph_nodes, sampled_2_intersect_nodes]))
                    current_N = subgraph_nodes.size(0)

                # Augmentation step 3 (intersection mode): 2-hop union (failsafe)
                if current_N < target_subgraph_size:
                    num_to_sample_2_union = target_subgraph_size - current_N

                    if cat_2_hop is None:
                        nodes_2_hop_u, _, _, _ = k_hop_subgraph(u, 2, data.edge_index, False)
                        nodes_2_hop_v, _, _, _ = k_hop_subgraph(v, 2, data.edge_index, False)
                        cat_2_hop = torch.cat([nodes_2_hop_u, nodes_2_hop_v])

                    nodes_2_hop_union = torch.unique(cat_2_hop)

                    is_already_present_mask_2_union = torch.isin(nodes_2_hop_union,
                                                                 subgraph_nodes)
                    candidate_nodes_2_union = nodes_2_hop_union[~is_already_present_mask_2_union]
                    num_candidates_2_union = candidate_nodes_2_union.size(0)

                    sampled_2_union_nodes = torch.tensor([], dtype=initial_subgraph_nodes.dtype, device=device)

                    if num_candidates_2_union > 0:
                        candidate_degrees_union = total_degree_tensor[candidate_nodes_2_union].float()
                        candidate_weights_union = 1.0 / (candidate_degrees_union + degree_epsilon)
                        candidate_weights_union.clamp_min_(1e-6)

                        actual_num_to_sample_union = min(num_to_sample_2_union, num_candidates_2_union)

                        if candidate_weights_union.sum() <= 1e-6:
                            perm_union = torch.randperm(num_candidates_2_union, device=device)
                            sampled_indices_union = perm_union[:actual_num_to_sample_union]
                        else:
                            try:
                                sampled_indices_union = torch.multinomial(candidate_weights_union,
                                                                          actual_num_to_sample_union, replacement=False)
                            except RuntimeError:
                                perm_union = torch.randperm(num_candidates_2_union, device=device)
                                sampled_indices_union = perm_union[:actual_num_to_sample_union]

                        sampled_2_union_nodes = candidate_nodes_2_union[sampled_indices_union]

                    subgraph_nodes = torch.unique(torch.cat([subgraph_nodes, sampled_2_union_nodes]))

            else:
                # Logic for 'union' mode
                # Priority: 2-hop intersect -> 2-hop union (failsafe)

                # Augmentation step 1: weighted sampling from 2-hop intersection
                num_to_sample_intersect = target_subgraph_size - N_initial

                nodes_2_hop_u, _, _, _ = k_hop_subgraph(u, 2, data.edge_index, False)
                nodes_2_hop_v, _, _, _ = k_hop_subgraph(v, 2, data.edge_index, False)
                cat_2_hop = torch.cat([nodes_2_hop_u, nodes_2_hop_v])

                uniq_2_hop, counts_2_hop = torch.unique(cat_2_hop, return_counts=True)
                nodes_2_hop_intersect = uniq_2_hop[counts_2_hop > 1]

                is_already_present_mask_intersect = torch.isin(nodes_2_hop_intersect, initial_subgraph_nodes)
                candidate_nodes_intersect = nodes_2_hop_intersect[~is_already_present_mask_intersect]
                num_candidates_intersect = candidate_nodes_intersect.size(0)

                sampled_intersect_nodes = torch.tensor([], dtype=initial_subgraph_nodes.dtype, device=device)

                if num_candidates_intersect > 0:
                    candidate_degrees_intersect = total_degree_tensor[candidate_nodes_intersect].float()
                    candidate_weights_intersect = 1.0 / (candidate_degrees_intersect + degree_epsilon)
                    candidate_weights_intersect.clamp_min_(1e-6)

                    actual_num_to_sample_intersect = min(num_to_sample_intersect, num_candidates_intersect)

                    if candidate_weights_intersect.sum() <= 1e-6:
                        perm = torch.randperm(num_candidates_intersect, device=device)
                        sampled_indices = perm[:actual_num_to_sample_intersect]
                    else:
                        try:
                            sampled_indices = torch.multinomial(candidate_weights_intersect,
                                                                actual_num_to_sample_intersect, replacement=False)
                        except RuntimeError as e:
                            perm = torch.randperm(num_candidates_intersect, device=device)
                            sampled_indices = perm[:actual_num_to_sample_intersect]

                    sampled_intersect_nodes = candidate_nodes_intersect[sampled_indices]

                subgraph_nodes = torch.unique(torch.cat([initial_subgraph_nodes, sampled_intersect_nodes]))
                N_after_intersect_sampling = subgraph_nodes.size(0)

                # Augmentation step 2 (failsafe): weighted sampling from 2-hop union
                if N_after_intersect_sampling < target_subgraph_size:
                    num_to_sample_union = target_subgraph_size - N_after_intersect_sampling

                    nodes_2_hop_union = torch.unique(cat_2_hop)

                    is_already_present_mask_union = torch.isin(nodes_2_hop_union,
                                                               subgraph_nodes)
                    candidate_nodes_union = nodes_2_hop_union[~is_already_present_mask_union]
                    num_candidates_union = candidate_nodes_union.size(0)

                    sampled_union_nodes = torch.tensor([], dtype=initial_subgraph_nodes.dtype,
                                                       device=device)

                    if num_candidates_union > 0:
                        candidate_degrees_union = total_degree_tensor[candidate_nodes_union].float()
                        candidate_weights_union = 1.0 / (candidate_degrees_union + degree_epsilon)
                        candidate_weights_union.clamp_min_(1e-6)

                        actual_num_to_sample_union = min(num_to_sample_union, num_candidates_union)

                        if candidate_weights_union.sum() <= 1e-6:
                            perm_union = torch.randperm(num_candidates_union, device=device)
                            sampled_indices_union = perm_union[:actual_num_to_sample_union]
                        else:
                            try:
                                sampled_indices_union = torch.multinomial(candidate_weights_union,
                                                                          actual_num_to_sample_union, replacement=False)
                            except RuntimeError as e:
                                perm_union = torch.randperm(num_candidates_union, device=device)
                                sampled_indices_union = perm_union[:actual_num_to_sample_union]

                        sampled_union_nodes = candidate_nodes_union[sampled_indices_union]

                        subgraph_nodes = torch.unique(torch.cat([subgraph_nodes, sampled_union_nodes]))

        # Gather nodes for the final subgraph
        N = subgraph_nodes.size(0)
        if N == 0:
            print(f"Warning: Skipping pair {i} ({u},{v}) - Final subgraph is empty.")
            continue

        is_mutual_interactor = torch.isin(subgraph_nodes, mutual_1_hop_nodes).int()

        # Subgraph extraction and feature processing
        subgraph_edge_index, subgraph_edge_type = subgraph(
            subset=subgraph_nodes,
            edge_index=data.edge_index,
            edge_attr=data.edge_type,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )

        if subgraph_edge_type is not None:
            subgraph_edge_type = subgraph_edge_type.long()

        adj_matrix = torch.full((N, N), num_edge_types, dtype=torch.long, device=device)
        if subgraph_edge_index.numel() > 0:
            edge_type_val = subgraph_edge_type if subgraph_edge_type is not None else 0
            if subgraph_edge_type is None:
                print(f"Warning: Edge types are None for pair {i}. Assigning default type 0.")
            adj_matrix[subgraph_edge_index[0], subgraph_edge_index[1]] = edge_type_val

        # Feature extraction
        try:
            path_sums_sub = path_sum_tensor[subgraph_nodes, :][:, subgraph_nodes]
            pairwise_dist_sub = spd_tensor[subgraph_nodes, :][:, subgraph_nodes]
            dist_to_u = spd_tensor[subgraph_nodes, u]
            dist_to_v = spd_tensor[subgraph_nodes, v]
            pairwise_dist_float = pairwise_dist_sub.float()
            average_edge_type_encoding = path_sums_sub.float() / (pairwise_dist_float + 1e-6)
            average_edge_type_encoding[pairwise_dist_float <= 0] = 0.0
            in_degrees_sub = in_degree_tensor[subgraph_nodes]
            out_degrees_sub = out_degree_tensor[subgraph_nodes]
            in_degrees_binned = bin_logarithmically(in_degrees_sub, max_in_degree, num_degree_bins)
            out_degrees_binned = bin_logarithmically(out_degrees_sub, max_out_degree, num_degree_bins)

        except IndexError as e:
            print(
                f"Error: IndexError during feature extraction for pair {i} ({u},{v}). Subgraph nodes might be out of bounds.")
            print(f"Subgraph nodes: {subgraph_nodes.tolist()}")
            print(
                f"Max index: spd={spd_tensor.shape[0]}, path={path_sum_tensor.shape[0]}, in_deg={in_degree_tensor.shape[0]}, out_deg={out_degree_tensor.shape[0]}")
            print(f"Skipping this pair due to error: {e}")
            continue
        except NameError:
            print("Error: bin_logarithmically function not defined.")
            raise
        except Exception as e:
            print(f"Error during feature extraction or binning for pair {i}: {e}")
            print(f"Skipping this pair.")
            continue

        # Post-process pairwise distances
        pairwise_dist_sub = pairwise_dist_sub.clone()
        dist_to_u = dist_to_u.clone()
        dist_to_v = dist_to_v.clone()

        pairwise_dist_sub[pairwise_dist_sub == -1] = default_dist
        pairwise_dist_sub.clamp_max_(default_dist)
        dist_to_u.clamp_max_(max_dist_uv)
        dist_to_v.clamp_max_(max_dist_uv)
        dist_to_u[dist_to_u == -1] = max_dist_uv + 1
        dist_to_v[dist_to_v == -1] = max_dist_uv + 1

        # Create 'node_perturbation_ids'
        # Based on the model: nn.Embedding(4, ..., padding_idx=3)
        padding_idx = 3
        N = subgraph_nodes.size(0)

        node_perturbation_ids = torch.full((N,), padding_idx, dtype=torch.long, device=device)
        u_local_idx = (subgraph_nodes == u).nonzero(as_tuple=True)[0]
        v_local_idx = (subgraph_nodes == v).nonzero(as_tuple=True)[0]

        # Update the tensor with the specific perturbation IDs for u and v
        node_perturbation_ids[u_local_idx] = pert_u
        node_perturbation_ids[v_local_idx] = pert_v

        # Build lifespan_genes dictionary
        lifespan_genes = torch.zeros(N, dtype=torch.bool, device=device)
        if hasattr(data, 'lifespan_association') and data.lifespan_association is not None:
            try:
                lifespan_genes = data.lifespan_association[subgraph_nodes]
            except IndexError:
                print(f"Warning: IndexError accessing lifespan_association for pair {i}. Using default (zeros).")

        # Build lifespan_distances dictionary
        lifespan_distances = torch.zeros(N, dtype=torch.bool, device=device)
        if hasattr(data, 'lifespan_dist') and data.lifespan_dist is not None:
            try:
                lifespan_distances = data.lifespan_dist[subgraph_nodes]
            except IndexError:
                print(f"Warning: IndexError accessing lifespan_dist for pair {i}. Using default (zeros).")

        sample = {
            'pair': pair_tensor,
            'pair_perturbations': pair_pert_tensor,
            'node_ids': subgraph_nodes,
            'node_perturbations': node_perturbation_ids,
            'dist_to_u': dist_to_u.to(torch.long),
            'dist_to_v': dist_to_v.to(torch.long),
            'adj_matrix': adj_matrix,
            'average_edge_type_encoding': average_edge_type_encoding,
            'pairwise_dist': pairwise_dist_sub.long(),
            'in_degree': in_degrees_sub.long(),
            'out_degree': out_degrees_sub.long(),
            'in_degree_binned': in_degrees_binned,
            'out_degree_binned': out_degrees_binned,
            'is_lifespan_gene': lifespan_genes,
            'is_mutual_interactor': is_mutual_interactor,
            'lifespan_dist': lifespan_distances,
        }

        if has_soft_targets:
            sample['target_soft'] = data.pair_effect_type_soft[i]
        if has_smoothed_targets:
            sample['target_soft_smoothed'] = data.pair_effect_type_soft_smoothed[i]

        processed_samples.append(sample)

    # Save subgraphs
    if output_path is not None:
        print(f"\nProcessing complete. Saving {len(processed_samples)} samples to {output_path}")
        try:
            torch.save(processed_samples, output_path)
            print(f"Processed samples saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving processed samples: {e}")
            raise
        print("--- Pre-processing finished successfully! ---")
    else:
        print(f"--- Pre-processing finished successfully! Returning {len(processed_samples)} samples. ---")
        return processed_samples


def process_data_to_soft_smoothed_label_dist(data, prior,
                                                 effect_type_key='effect_type',
                                                 pair_index_key='pair_index',
                                                 pair_pert_index_key='pair_pert_index'):
    """
    Processes a PyG Data object with repeated (gene-pert, gene-pert) pairs and
    raw labels into a new Data object with unique pairs and smoothed labels.

    This version groups observations by their unique (gene1, pert1, gene2, pert2)
    tuple, treating (A, B) as identical to (B, A). It then uses a
    Dirichlet-Multinomial model to smooth the labels, incorporating a prior.
    """
    # 1. Get data and validate inputs
    if (pair_index_key not in data or
            pair_pert_index_key not in data or
            effect_type_key not in data):
        raise ValueError(
            f"Data object must contain '{pair_index_key}', "
            f"'{pair_pert_index_key}', and '{effect_type_key}'"
        )

    pair_index = data[pair_index_key]
    pair_pert_index = data[pair_pert_index_key]
    pair_labels = data[effect_type_key]
    num_classes = int(pair_labels.max().item() + 1)
    original_device = pair_labels.device

    if not isinstance(prior, torch.Tensor):
        raise TypeError("Prior must be a torch.Tensor.")
    if prior.shape != (num_classes,):
        raise ValueError(f"Prior shape must be ({num_classes},), but got {prior.shape}.")

    # 2. Create canonical (gene-pert, gene-pert) grouping keys.
    #    We want to group by (g1, p1, g2, p2) and treat
    #    (g1, p1, g2, p2) as identical to (g2, p2, g1, p1).

    # Create [N_obs, 2] tensors for the first and second part of the pair.
    # pair_A = (g1, p1) for all obs
    # pair_B = (g2, p2) for all obs
    pair_A = torch.stack([pair_index[0], pair_pert_index[0]], dim=1)
    pair_B = torch.stack([pair_index[1], pair_pert_index[1]], dim=1)

    # Find where to swap. Swap if g1 > g2, or if g1 == g2 and p1 > p2.
    swap_mask = (pair_A[:, 0] > pair_B[:, 0]) | \
                ((pair_A[:, 0] == pair_B[:, 0]) & (pair_A[:, 1] > pair_B[:, 1]))

    # Create a [N_obs, 4] tensor to hold the keys
    sorted_pair_A = torch.where(swap_mask.unsqueeze(1), pair_B, pair_A)
    sorted_pair_B = torch.where(swap_mask.unsqueeze(1), pair_A, pair_B)

    # group_keys_sorted is [N_obs, 4], where each row is (g1, p1, g2, p2)
    group_keys_sorted = torch.cat([sorted_pair_A, sorted_pair_B], dim=1)

    # 3. Find unique groups and get their raw label counts.
    unique_groups, inverse_indices = torch.unique(
        group_keys_sorted.cpu(), dim=0, return_inverse=True
    )
    unique_groups = unique_groups.to(original_device)
    inverse_indices = inverse_indices.to(original_device)

    num_unique_groups = unique_groups.shape[0]
    counts = torch.zeros(num_unique_groups, num_classes, dtype=torch.float, device=original_device)

    index = inverse_indices.unsqueeze(1).expand(-1, num_classes)
    one_hot_labels = F.one_hot(pair_labels, num_classes=num_classes).float()
    counts.scatter_add_(0, index, one_hot_labels)

    # Calculate unsmoothed soft labels (relative frequencies)
    total_counts = counts.sum(dim=1, keepdim=True)
    total_counts_clamped = torch.clamp(total_counts, min=1)
    soft_labels = counts / total_counts_clamped

    # 4. Apply Bayesian Smoothing (Dirichlet-Multinomial)
    prior_on_device = prior.to(original_device).float()

    # Posterior parameters are alpha' = alpha + c
    posterior_params = counts + prior_on_device

    # The expected value of the posterior is E[p_i] = alpha'_i / sum(alpha')
    posterior_sum = posterior_params.sum(dim=1, keepdim=True)
    posterior_sum = torch.clamp(posterior_sum, min=1e-9)
    smoothed_labels = posterior_params / posterior_sum

    # 5. Create the new, clean Data object.
    new_data = Data()

    keys_to_exclude = {pair_index_key, pair_pert_index_key, effect_type_key}

    for key, value in data:
        if key not in keys_to_exclude:
            new_data[key] = value

    # Add the newly created unique groups, smoothed labels, and counts
    new_data.pair_pert_group_index = unique_groups.T
    new_data.pair_effect_type_soft = soft_labels
    new_data.pair_effect_type_soft_smoothed = smoothed_labels
    new_data.pair_obs_counts = counts
    new_data.pair_obs_total_counts = total_counts.squeeze()

    return new_data


def graphormer_collate_fn(batch, pad_values):
    """
    A robust collate function that batches graph data by checking tensor dimensions.
    It correctly handles variable-length 1D node features and 2D pairwise matrices.
    """
    # Filter out any potential None samples returned by the dataset
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # 1. Determine max node count and create the node padding mask
    max_len = max(sample['node_ids'].size(0) for sample in batch)
    padded_batch = {}

    padding_masks = []
    for sample in batch:
        num_nodes = sample['node_ids'].size(0)
        pad_len = max_len - num_nodes
        # Mask is False for real nodes, True for padding
        padding_masks.append(torch.cat([
            torch.zeros(num_nodes, dtype=torch.bool),
            torch.ones(pad_len, dtype=torch.bool)
        ]))
    padded_batch['padding_mask'] = torch.stack(padding_masks)

    # 2. Iterate through all keys and batch them according to their shape
    all_keys = list(batch[0].keys())

    for key in all_keys:
        tensors_to_batch = [sample[key] for sample in batch]
        pad_value = pad_values.get(key, 0)
        first_tensor = tensors_to_batch[0]

        # Manual padding for 2D matrices
        if first_tensor.dim() == 2:
            padded_list = [
                F.pad(t, (0, max_len - t.size(1), 0, max_len - t.size(0)), value=pad_value)
                for t in tensors_to_batch
            ]
            padded_batch[key] = torch.stack(padded_list)

        # Padding for 1D vectors
        elif first_tensor.dim() == 1:
            if key in ['pair', 'pair_perturbations']:
                padded_batch[key] = torch.stack(tensors_to_batch)
            else:
                padded_batch[key] = pad_sequence(tensors_to_batch, batch_first=True, padding_value=pad_value)

        # Padding for scalars or others
        else:
            padded_batch[key] = torch.stack(tensors_to_batch)

    return padded_batch

