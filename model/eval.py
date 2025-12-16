import torch
import numpy as np
from scipy.stats import kendalltau

def ndcg_score(y_preds: torch.Tensor, y_true: torch.Tensor, k: int = None):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) for a batch.

    Args:
        y_preds (torch.Tensor): Predicted distributions. Shape: [batch_size, num_classes]
        y_true (torch.Tensor): True distributions (relevances). Shape: [batch_size, num_classes]
        k (int, optional): The number of top results to consider (e.g., NDCG@10).
                           If None, all results are used. Defaults to None.

    Returns:
        torch.Tensor: The average NDCG score for the batch.
    """
    if k is None:
        k = y_true.size(1)

    _, top_indices = torch.topk(y_preds, k, dim=1)

    relevances = torch.gather(y_true, 1, top_indices)
    discounts = torch.log2(torch.arange(2, k + 2).float()).to(y_preds.device)
    dcg = (relevances / discounts).sum(dim=1)

    ideal_relevances, _ = torch.topk(y_true, k, dim=1)
    idcg = (ideal_relevances / discounts).sum(dim=1)

    ndcg = dcg / (idcg + 1e-8) # Add epsilon for stability

    return ndcg.mean()

def top_k_set_agreement(y_preds: torch.Tensor, y_true: torch.Tensor, k: int):
    """
    Computes the Top-k Set Agreement score for a batch.

    This metric measures the proportion of samples where the set of the top-k
    predicted label indices is exactly the same as the set of the top-k
    true label indices.

    Args:
        y_preds (torch.Tensor): Predicted distributions. Shape: [batch_size, num_classes]
        y_true (torch.Tensor): True distributions. Shape: [batch_size, num_classes]
        k (int): The number of top labels to include in the set.

    Returns:
        torch.Tensor: The average agreement score (a float between 0.0 and 1.0).
    """
    _, pred_indices = torch.topk(y_preds, k, dim=1)
    _, true_indices = torch.topk(y_true, k, dim=1)

    pred_indices_sorted, _ = torch.sort(pred_indices, dim=1)
    true_indices_sorted, _ = torch.sort(true_indices, dim=1)

    matches = (pred_indices_sorted == true_indices_sorted).all(dim=1)

    agreement_rate = matches.float().mean()

    return agreement_rate

def kendalls_tau_score(y_preds: torch.Tensor, y_true: torch.Tensor):
    """
    Computes the average Kendall's Tau rank correlation for a batch.

    This function uses the SciPy library to calculate Kendall's Tau for each
    sample in the batch and then returns the average score.

    Args:
        y_preds (torch.Tensor): Predicted distributions. Shape: [batch_size, num_classes]
        y_true (torch.Tensor): True distributions. Shape: [batch_size, num_classes]

    Returns:
        float: The average Kendall's Tau score for the batch.
    """
    y_preds_np = y_preds.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    tau_scores = []
    for i in range(y_preds_np.shape[0]):
        tau_result = kendalltau(y_preds_np[i], y_true_np[i])
        tau_scores.append(tau_result.correlation)

    return np.nanmean(tau_scores)

def per_class_top_1_agreement(y_preds: torch.Tensor, y_true: torch.Tensor):
    """
    Computes the Top-1 agreement for each class individually.

    For each class `c`, this metric calculates the fraction of samples where
    the model correctly predicts `c` as the top class, considering only the
    subset of samples where `c` is the true top class.

    Args:
        y_preds (torch.Tensor): Predicted distributions. Shape: [batch_size, num_classes]
        y_true (torch.Tensor): True distributions. Shape: [batch_size, num_classes]

    Returns:
        dict: A dictionary where keys are class indices and values are the
              Top-1 agreement scores for that class. Classes that are never
              the true top class in the batch are omitted from the dictionary.
    """
    if y_preds.dim() != 2 or y_true.dim() != 2:
        raise ValueError("Input tensors must be 2-dimensional.")
    if y_preds.shape != y_true.shape:
        raise ValueError("Input tensors must have the same shape.")

    pred_top_indices = torch.argmax(y_preds, dim=1)
    true_top_indices = torch.argmax(y_true, dim=1)

    agreement_scores = {}
    num_classes = y_true.size(1)

    for c in range(num_classes):
        true_mask = (true_top_indices == c)

        total_for_class = torch.sum(true_mask)

        if total_for_class == 0:
            continue

        preds_for_subset = pred_top_indices[true_mask]

        correct_predictions = torch.sum(preds_for_subset == c)

        agreement = correct_predictions.float() / total_for_class.float()
        agreement_scores[c] = agreement.item()

    return agreement_scores