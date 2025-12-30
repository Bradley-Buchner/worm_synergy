import torch.nn.functional as F
from torchmetrics import AUROC, AveragePrecision
from tqdm import tqdm
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, norm
import random
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from collections import defaultdict
import copy
from functools import partial
from torch.utils.data import DataLoader
import networkx as nx
from torch_geometric.utils import subgraph, degree
from itertools import combinations, product, chain


from model.eval import *
from model.models import NaiveBaseline, UniformBaseline


def train_test_split_simple(processed_data, train_frac=0.8, seed=23):
    """
    Splits a list of processed data into training and testing subsets using a random shuffle.

    Params:
    ------
    processed_data : list
        A list of data objects (e.g., PyG Data objects) where each item represents a unique gene pair.
    train_frac : float, optional
        The fraction of data to include in the training set (default is 0.8).
    seed : int, optional
        Random seed for reproducibility (default is 23).

    Returns:
    --------
    train_data : list
        The subset of data for model training.
    test_data : list
        The subset of data for model evaluation.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    shuffled_data = processed_data[:]
    random.shuffle(shuffled_data)

    split_idx = int(len(shuffled_data) * train_frac)
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]

    return train_data, test_data


def train_epoch_trans(model, dataloader, label_name, optimizer, criterion, device, randomize_labels=False):
    """
    Evaluates the model on a validation or test set using Label Distribution Learning (LDL) metrics.

    Params:
    ------
    model : torch.nn.Module
        The SynergyGT model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The evaluation data loader.
    label_name : str
        The key in the batch dictionary corresponding to the target labels.
    criterion : callable
        The loss function used for evaluation.
    device : torch.device or str
        The device to run evaluation on.
    metrics : dict, optional
        A dictionary of torchmetrics (e.g., AUROC) to update during the epoch.

    Returns:
    -------
    epoch_loss : float
        The average loss over the evaluation set.
    results : dict
        A dictionary containing calculated metrics including Log-Loss, Cosine Similarity,
        NDCG, and per-class agreement scores.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch in tqdm(dataloader, desc="Train"):
        labels = batch.pop(label_name).to(device)
        if randomize_labels:
            random_indices = torch.randperm(labels.size(0))
            labels = labels[random_indices]
        inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)
            else:
                raise TypeError(
                    f"All batch values must be torch.Tensors, but key '{key}' has type {type(value).__name__}"
                )

        optimizer.zero_grad()
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=1)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss


def eval_epoch_trans(model, dataloader, label_name, criterion, device, metrics={}):
    """
    Evaluates the model and optionally generates a scatterplot of predicted vs. actual synergy.

    Params:
    ------
    model : torch.nn.Module
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The evaluation data loader.
    label_name : str
        The target label key.
    criterion : callable
        The evaluation loss function.
    device : torch.device or str
        The computing device.
    metrics : dict, optional
        Torchmetrics to calculate.
    plot_fig : bool, optional
        If True, generates and displays a scatterplot of synergy probabilities (default is False).

    Returns:
    -------
    epoch_loss : float
        The average evaluation loss.
    results : dict
        Dictionary of calculated performance metrics.
    predictions : dict
        Dictionary containing 'preds' and 'labels' for the entire dataset.
    """
    model.eval()
    running_loss = 0.0
    running_log_loss = 0.0
    running_cosine_sim = 0.0
    running_ndcg = 0.0
    running_agreement_k1 = 0.0
    running_agreement_k2 = 0.0
    running_kendalls = 0.0
    total_samples = 0
    per_class_correct_counts = {}
    per_class_total_counts = {}
    correct_for_0_or_2 = 0
    total_for_0_or_2 = 0

    if metrics:
        for metric in metrics.values():
            metric.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            labels = batch.pop(label_name).to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(inputs)
            preds_probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, labels)
            log_loss = -torch.sum(labels * log_probs, dim=1).mean()

            # Binarize targets for multiclass metrics
            target_binary = torch.argmax(labels, dim=1)

            if metrics:
                for metric in metrics.values():
                    metric.update(preds_probs, target_binary)

            # Other LDL-specific metrics
            running_loss += loss.item() * labels.size(0)
            running_log_loss += log_loss.item() * labels.size(0)
            cosine_sim_batch = F.cosine_similarity(preds_probs, labels, dim=1)
            running_cosine_sim += torch.sum(cosine_sim_batch).item()
            ndcg_batch = ndcg_score(preds_probs, labels)
            running_ndcg += ndcg_batch.item() * labels.size(0)
            agreement_k1_batch = top_k_set_agreement(preds_probs, labels, k=1)
            agreement_k2_batch = top_k_set_agreement(preds_probs, labels, k=2)
            running_agreement_k1 += agreement_k1_batch.item() * labels.size(0)
            running_agreement_k2 += agreement_k2_batch.item() * labels.size(0)
            kendalls_batch = kendalls_tau_score(preds_probs, labels)
            running_kendalls += kendalls_batch * labels.size(0)
            total_samples += labels.size(0)

            # per-class and subset agreement metric logic
            preds_binary = torch.argmax(preds_probs, dim=1)
            num_classes = labels.size(1)
            for c in range(num_classes):
                true_mask = (target_binary == c)
                total_for_class_in_batch = torch.sum(true_mask)
                if total_for_class_in_batch > 0:
                    correct_in_batch = torch.sum(preds_binary[true_mask] == c)
                    per_class_correct_counts[c] = per_class_correct_counts.get(c, 0) + correct_in_batch.item()
                    per_class_total_counts[c] = per_class_total_counts.get(c, 0) + total_for_class_in_batch.item()
            subset_mask = (target_binary == 0) | (target_binary == 2)
            batch_total_subset = torch.sum(subset_mask)
            if batch_total_subset > 0:
                batch_correct_subset = torch.sum(preds_binary[subset_mask] == target_binary[subset_mask])
                total_for_0_or_2 += batch_total_subset.item()
                correct_for_0_or_2 += batch_correct_subset.item()

    # Final results
    results = {name: metric.compute() for name, metric in metrics.items()}

    for name, metric in results.items():
        if name in ["auc_per_class", "auprc_per_class"]:
            results[name] = [round(v.item(), 4) for v in metric]
        elif name in ["auc", "auprc"]:
            results[name] = round(metric.item(), 4)

    epoch_loss = running_loss / total_samples
    epoch_log_loss = running_log_loss / total_samples
    results['log_loss'] = round(epoch_log_loss, 4)
    results["cosine_similarity"] = round(running_cosine_sim / total_samples, 4)
    results["ndcg"] = round(running_ndcg / total_samples, 4)
    results["agreement_k1"] = round(running_agreement_k1 / total_samples, 4)
    results["agreement_k2"] = round(running_agreement_k2 / total_samples, 4)
    results["kendalls"] = round(running_kendalls / total_samples, 4)

    epoch_per_class_agreement = {
        c: per_class_correct_counts.get(c, 0) / per_class_total_counts[c]
        for c in per_class_total_counts
    }
    results["per_class_top_1_agreement"] = np.round(np.array(list(epoch_per_class_agreement.values())), 4)
    agreement_0_or_2 = (correct_for_0_or_2 / total_for_0_or_2) if total_for_0_or_2 > 0 else 0.0
    results["agreement_top1_when_0_or_2"] = round(agreement_0_or_2, 4)

    return epoch_loss, results


def eval_epoch_trans_plt(model, dataloader, label_name, criterion, device, metrics={}, plot_fig=False):
    """
    Evaluates the model and optionally generates a scatterplot of predicted vs. actual synergy.

    Params:
    ------
    model : torch.nn.Module
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The evaluation data loader.
    label_name : str
        The target label key.
    criterion : callable
        The evaluation loss function.
    device : torch.device or str
        The computing device.
    metrics : dict, optional
        Torchmetrics to calculate.
    plot_fig : bool, optional
        If True, generates and displays a scatterplot of synergy probabilities (default is False).

    Returns:
    -------
    epoch_loss : float
        The average evaluation loss.
    results : dict
        Dictionary of calculated performance metrics.
    predictions : dict
        Dictionary containing 'preds' and 'labels' for the entire dataset.
    """
    model.eval()
    running_loss = 0.0
    running_log_loss = 0.0
    running_cosine_sim = 0.0
    running_ndcg = 0.0
    running_agreement_k1 = 0.0
    running_agreement_k2 = 0.0
    running_kendalls = 0.0
    total_samples = 0
    per_class_correct_counts = {}
    per_class_total_counts = {}
    correct_for_0_or_2 = 0
    total_for_0_or_2 = 0

    all_synergy_preds = []
    all_synergy_labels = []
    all_preds = []
    all_labels = []

    if metrics:
        for metric in metrics.values():
            metric.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            labels = batch.pop(label_name).to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(inputs)
            preds_probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, labels)
            log_loss = -torch.sum(labels * log_probs, dim=1).mean()

            # Binarize targets for multiclass metrics
            target_binary = torch.argmax(labels, dim=1)

            if metrics:
                for metric in metrics.values():
                    metric.update(preds_probs, target_binary)

            # Collect data for scatterplot
            if plot_fig:
                synergy_col_idx = 2
                all_synergy_preds.extend(preds_probs[:, synergy_col_idx].cpu().numpy())
                all_synergy_labels.extend(labels[:, synergy_col_idx].cpu().numpy())
            all_preds.extend(preds_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Other LDL-specific metrics
            running_loss += loss.item() * labels.size(0)
            running_log_loss += log_loss.item() * labels.size(0)
            cosine_sim_batch = F.cosine_similarity(preds_probs, labels, dim=1)
            running_cosine_sim += torch.sum(cosine_sim_batch).item()
            ndcg_batch = ndcg_score(preds_probs, labels)
            running_ndcg += ndcg_batch.item() * labels.size(0)
            agreement_k1_batch = top_k_set_agreement(preds_probs, labels, k=1)
            agreement_k2_batch = top_k_set_agreement(preds_probs, labels, k=2)
            running_agreement_k1 += agreement_k1_batch.item() * labels.size(0)
            running_agreement_k2 += agreement_k2_batch.item() * labels.size(0)
            kendalls_batch = kendalls_tau_score(preds_probs, labels)
            running_kendalls += kendalls_batch * labels.size(0)
            total_samples += labels.size(0)

            # Per-class and subset agreement metric logic
            preds_binary = torch.argmax(preds_probs, dim=1)
            num_classes = labels.size(1)
            for c in range(num_classes):
                true_mask = (target_binary == c)
                total_for_class_in_batch = torch.sum(true_mask)
                if total_for_class_in_batch > 0:
                    correct_in_batch = torch.sum(preds_binary[true_mask] == c)
                    per_class_correct_counts[c] = per_class_correct_counts.get(c, 0) + correct_in_batch.item()
                    per_class_total_counts[c] = per_class_total_counts.get(c, 0) + total_for_class_in_batch.item()
            subset_mask = (target_binary == 0) | (target_binary == 2)
            batch_total_subset = torch.sum(subset_mask)
            if batch_total_subset > 0:
                batch_correct_subset = torch.sum(preds_binary[subset_mask] == target_binary[subset_mask])
                total_for_0_or_2 += batch_total_subset.item()
                correct_for_0_or_2 += batch_correct_subset.item()

    # Final results
    results = {name: metric.compute() for name, metric in metrics.items()}

    for name, metric in results.items():
        if name in ["auc_per_class", "auprc_per_class"]:
            results[name] = [round(v.item(), 4) for v in metric]
        elif name in ["auc", "auprc"]:
            results[name] = round(metric.item(), 4)

    epoch_loss = running_loss / total_samples
    epoch_log_loss = running_log_loss / total_samples
    results['log_loss'] = round(epoch_log_loss, 4)
    results["cosine_similarity"] = round(running_cosine_sim / total_samples, 4)
    results["ndcg"] = round(running_ndcg / total_samples, 4)
    results["agreement_k1"] = round(running_agreement_k1 / total_samples, 4)
    results["agreement_k2"] = round(running_agreement_k2 / total_samples, 4)
    results["kendalls"] = round(running_kendalls / total_samples, 4)

    epoch_per_class_agreement = {
        c: per_class_correct_counts.get(c, 0) / per_class_total_counts[c]
        for c in per_class_total_counts
    }
    results["per_class_top_1_agreement"] = np.round(np.array(list(epoch_per_class_agreement.values())), 4)
    agreement_0_or_2 = (correct_for_0_or_2 / total_for_0_or_2) if total_for_0_or_2 > 0 else 0.0
    results["agreement_top1_when_0_or_2"] = round(agreement_0_or_2, 4)

    predictions = {"preds": all_preds, "labels": all_labels}

    # Scatterplot generation
    if plot_fig:
        synergy_preds_arr = np.array(all_synergy_preds)
        synergy_labels_arr = np.array(all_synergy_labels)

        corr, _ = pearsonr(synergy_labels_arr, synergy_preds_arr)

        plt.figure(figsize=(8, 8))
        plt.scatter(synergy_labels_arr, synergy_preds_arr, alpha=0.4, color='blue', s=10, label='Test Data')

        lims = [
            np.min([plt.xlim(), plt.ylim()]),  # min of both axes
            np.max([plt.xlim(), plt.ylim()]),  # max of both axes
        ]
        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal Prediction')

        plt.title(f'Synergy Probability: Actual vs Predicted\nPearson r = {corr:.4f}')
        plt.xlabel('Actual Synergy Probability (Soft Label)')
        plt.ylabel('Predicted Synergy Probability')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # Save or show plot
        # plt.savefig('synergy_scatterplot.png', dpi=300)
        # print(f"Scatterplot saved to synergy_scatterplot.png. Correlation r={corr:.4f}")
        plt.show()

    return epoch_loss, results, predictions


def train_synergy_model(model, train_loader, test_loader, label_name, optimizer, scheduler, loss_fn, randomize_labels,
                device, num_epochs=10, num_classes=3, seed=23):
    """
    High-level wrapper to train the SynergyGT model and evaluate it on a test set.

    Params:
    ------
    model : torch.nn.Module
        The model instance to train.
    train_loader : torch.utils.data.DataLoader
        The training data.
    test_loader : torch.utils.data.DataLoader or None
        The test data for evaluation. If None, evaluation is skipped.
    label_name : str
        The target label key.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler or None
        The learning rate scheduler.
    loss_fn : callable
        The loss function.
    randomize_labels : bool
        Whether to shuffle labels for baseline testing.
    device : torch.device or str
        The computing device.
    num_epochs : int, optional
        Number of training epochs (default is 10).
    num_classes : int, optional
        Number of interaction classes (default is 3).
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    -------
    results : dict or None
        The final test metrics.
    test_preds_final : dict or None
        The final predictions on the test set.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1:03d}")
        train_loss_epoch = train_epoch_trans(model, train_loader, label_name, optimizer, loss_fn, device,
                                             randomize_labels=randomize_labels)

        if test_loader is not None:
            test_loss_epoch, *_ = eval_epoch_trans_plt(model, test_loader, label_name, loss_fn, device,
                                                      metrics={}, plot_fig=False)
        else:
            test_loss_epoch = None

        model.history["train_loss"].append(train_loss_epoch)
        model.history["test_loss"].append(test_loss_epoch)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"lr={current_lr}  train_loss={train_loss_epoch:.4f}  "
              f"test_loss={test_loss_epoch:.4f}\n")

    print(f"Training complete!")

    if test_loader is not None:
        metrics_test_soft = {
            "auc": AUROC(task="multiclass", num_classes=num_classes, average="macro").to(device),
            "auc_per_class": AUROC(task="multiclass", num_classes=num_classes, average="none").to(device),
            "auprc": AveragePrecision(task="multiclass", num_classes=num_classes, average="macro").to(device),
            "auprc_per_class": AveragePrecision(task="multiclass", num_classes=num_classes, average="none").to(device)
        }


        test_loss_final, test_metrics_final, test_preds_final = eval_epoch_trans_plt(model, test_loader, label_name, loss_fn, device,
                                                                   metrics=metrics_test_soft, plot_fig=False)
        results = {name: m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m for name, m in
                   test_metrics_final.items()}
        results["loss_kl"] = test_loss_final

        model.history["test_metrics"].append(results)
    else:
        print("No test set provided, skipping evaluation.")
        results = None
        test_preds_final = None

    return results, test_preds_final


def get_predictions_synergy_model(model, dataloader, label_name=None, device='cpu'):
    """
    Runs model inference on a dataloader to generate predictions for gene pairs.

    Params:
    -------
    model : torch.nn.Module
        The trained SynergyGT model.
    dataloader : torch.utils.data.DataLoader
        The data to perform inference on.
    label_name : str, optional
        If provided, extracts and returns the true labels alongside predictions.
    device : torch.device or str, optional
        The computing device (default is 'cpu').

    Returns
    -------
    predictions : dict
        A dictionary containing 'pairs', 'preds', and optionally 'labels'.
    """
    model.eval()
    all_pairs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            if label_name is not None:
                labels = batch.pop(label_name).to(device)
            pairs = batch.pop('pair').to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(inputs)
            preds_probs = F.softmax(outputs, dim=1)

            all_pairs.extend(pairs.cpu().numpy())
            all_preds.extend(preds_probs.cpu().numpy())
            if label_name is not None:
                all_labels.extend(labels.cpu().numpy())

    predictions = {"pairs": all_pairs, "preds": all_preds, "labels": all_labels}

    return predictions


def enable_dropout(model):
    """
    Forces Dropout and BatchNorm layers into training mode to enable stochastic inference.

    Params:
    -------
    model : torch.nn.Module
        The model to modify.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout') or \
                m.__class__.__name__.startswith('BatchNorm'):
            m.train()


def predict_with_mc_dropout(model, dataloader, num_samples=100, device='cpu'):
    """
    Performs Monte Carlo Dropout inference to estimate prediction uncertainty.

    Params:
    -------
    model : torch.nn.Module
        The SynergyGT model.
    dataloader : torch.utils.data.DataLoader
        The dataloader containing gene pairs for inference.
    num_samples : int, optional
        The number of stochastic forward passes to perform (default is 100).
    device : torch.device or str, optional
        The computing device (default is 'cpu').

    Returns:
    -------
    all_results : list
        A list of dictionaries containing mean, std, 95% CI, and MAP estimates for each pair.
    stacked_preds : torch.Tensor
        A tensor of shape (num_samples, total_pairs, num_classes) containing all raw samples.
    """
    # Move model to device and set to eval
    model.to(device)
    model.eval()

    # Keep Dropout layers active during inference
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    # List to store full-dataset predictions for each stochastic pass
    all_sample_tensors = []

    # We iterate by sample first to make stacking easy (like get_mc_dropout_predictions)
    for _ in tqdm(range(num_samples), desc="MC Dropout Sampling"):
        batch_probs = []
        with torch.no_grad():
            for batch in dataloader:
                # SynergyGT inputs usually come in a dict or batch object
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs.cpu())

        # Concatenate all batches for this specific MC pass
        # Resulting shape: (total_pairs, 3)
        all_sample_tensors.append(torch.cat(batch_probs, dim=0))

    # Stack into a single tensor: (num_samples, total_pairs, 3)
    stacked_preds = torch.stack(all_sample_tensors, dim=0)

    # For statistical processing, convert to numpy once
    all_preds_np = stacked_preds.numpy()
    total_pairs = all_preds_np.shape[1]
    all_results = []

    # Calculate statistics for each pair
    for i in range(total_pairs):
        pair_samples = all_preds_np[:, i, :]  # (num_samples, 3)
        class_stats = []

        for c in range(3):
            samples_c = pair_samples[:, c]

            # Descriptive stats
            mean = np.mean(samples_c)
            std = np.std(samples_c)
            ci_lower, ci_upper = np.percentile(samples_c, [2.5, 97.5])

            # Using your existing MAP estimation utility
            map_estimate = estimate_map(samples_c)

            class_stats.append({
                'mean': mean,
                'std': std,
                'ci_95': (ci_lower, ci_upper),
                'map': map_estimate,
                'raw_samples': samples_c
            })

        all_results.append(class_stats)

    return all_results, stacked_preds


def get_synergy_model_performance(model_metrics, uninformative_baseline_metrics, mean_baseline_metrics):
    """
    Prints a formatted comparison of the SynergyGT model against uninformative and mean baselines.

    Params:
    ------
    model_metrics : dict
        Performance metrics for the trained SynergyGT model.
    uninformative_baseline_metrics : dict
        Metrics for a model predicting a uniform distribution.
    mean_baseline_metrics : dict
        Metrics for a model predicting the average training distribution.

    Returns:
    -------
    None
    """
    overall_data = {
        "Model": ["Synergy GT", "Uninformative Baseline", "Mean Baseline"],
        "AUROC (macro)": [
            model_metrics['auc'],
            uninformative_baseline_metrics['auc'],
            mean_baseline_metrics['auc']
        ],
        "AUPRC (macro)": [
            model_metrics['auprc'],
            uninformative_baseline_metrics['auprc'],
            mean_baseline_metrics['auprc']
        ],
        "Avg. KL Divergence": [
            model_metrics['loss_kl'],
            uninformative_baseline_metrics['loss_kl'],
            mean_baseline_metrics['loss_kl']
        ]
        # "Cosine Similarity": [
        #     model_metrics['cosine_similarity'],
        #     uninformative_baseline_metrics['cosine_similarity'],
        #     mean_baseline_metrics['cosine_similarity']
        # ]

    }
    df_overall = pd.DataFrame(overall_data)

    auroc_class_data = {
        "Antagonistic": [
            model_metrics['auc_per_class'][0],
            uninformative_baseline_metrics['auc_per_class'][0],
            mean_baseline_metrics['auc_per_class'][0]
        ],
        "Neither": [
            model_metrics['auc_per_class'][1],
            uninformative_baseline_metrics['auc_per_class'][1],
            mean_baseline_metrics['auc_per_class'][1]
        ],
        "Synergistic": [
            model_metrics['auc_per_class'][2],
            uninformative_baseline_metrics['auc_per_class'][2],
            mean_baseline_metrics['auc_per_class'][2]
        ]
    }
    auroc_df_per_class = pd.DataFrame(auroc_class_data, index=["Synergy GT", "Uninformative Baseline", "Mean Baseline"])

    auprc_class_data = {
        "Antagonistic": [
            model_metrics['auprc_per_class'][0],
            uninformative_baseline_metrics['auprc_per_class'][0],
            mean_baseline_metrics['auprc_per_class'][0]
        ],
        "Neither": [
            model_metrics['auprc_per_class'][1],
            uninformative_baseline_metrics['auprc_per_class'][1],
            mean_baseline_metrics['auprc_per_class'][1]
        ],
        "Synergistic": [
            model_metrics['auprc_per_class'][2],
            uninformative_baseline_metrics['auprc_per_class'][2],
            mean_baseline_metrics['auprc_per_class'][2]
        ]
    }
    auprc_df_per_class = pd.DataFrame(auprc_class_data, index=["Synergy GT", "Uninformative Baseline", "Mean Baseline"])

    print("\n=== Test Set Model Performance ===")
    print("\n--- Overall Comparison---")
    print(df_overall.round(2).to_string(index=False))

    print("\n--- Per-Class AUROC Comparison ---")
    print(auroc_df_per_class.round(2))

    print("\n--- Per-Class AUPRC Comparison ---")
    print(auprc_df_per_class.round(2))


def get_loader_inference(data_orig, pair_tuples, preprocessor_fn, preprocessor_fn_configs, dataset_cls,
                            collate_fn, collate_fn_configs, batch_size=32):
    """
    Initializes a DataLoader for inference on a custom list of gene pairs.

    Params:
    -------
    data_orig : torch_geometric.data.Data
        The original graph data object.
    pair_tuples : list of tuples
        List of (gene_A, gene_B) IDs to evaluate.
    preprocessor_fn : callable
        Function to extract subgraphs or features for the model.
    preprocessor_fn_configs : dict
        Configuration parameters for the preprocessor.
    dataset_cls : type
        The PyTorch Dataset class to use.
    collate_fn : callable
        The function to collate samples into batches.
    collate_fn_configs : dict
        Padding and configuration for the collate function.
    batch_size : int, optional
        The batch size for inference (default is 32).

    Returns
    -------
    loader : torch.utils.data.DataLoader
        The prepared data loader.
    subgraph_data : list
        The list of preprocessed subgraph objects.
    """

    print(f"\n--- Initializing Inference for {len(pair_tuples)} Pair(s) ---")

    data_inference = copy.copy(data_orig)
    # u_list, pu_list, v_list, pv_list = zip(*pair_tuples)
    u_list, v_list = zip(*pair_tuples)

    for u, v in zip(u_list, v_list):
        if u == v:
            raise ValueError(f"Identical gene pair detected (ID: {u}). "
                             f"Synergy inference requires two different genes.")

    pu_list = [1] * len(u_list)
    pv_list = [0] * len(v_list)
    pair_pert_tensor = torch.tensor([u_list, pu_list, v_list, pv_list], dtype=torch.long)

    data_inference.pair_pert_group_index = pair_pert_tensor
    data_inference.pair_index = torch.tensor([u_list, v_list], dtype=torch.long)

    # Nullify targets to ensure no leakage/errors during preprocessing
    if hasattr(data_inference, 'pair_effect_type_soft'):
        data_inference.pair_effect_type_soft = None
    if hasattr(data_inference, 'pair_effect_type_soft_smoothed'):
        data_inference.pair_effect_type_soft_smoothed = None

    # Run preprocessing
    subgraph_data = preprocessor_fn(
        data=data_inference,
        configs=preprocessor_fn_configs
    )

    if not subgraph_data:
        print("Preprocessing returned no samples. Check connectivity/indices.")
        return {}

    # DataLoader setup
    dataset = dataset_cls(subgraph_data)

    collate_partial = partial(collate_fn, pad_values=collate_fn_configs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_partial)

    return loader, subgraph_data


def get_unknown_gene_pairs(data, sampling_pool_node_ids, N_pool_pairs=None, pool_strategy='all',
                             min_mutual_interactors=0):
    """
    Samples gene pairs from the unlabeled pool based on a specific biological strategy.

    Params:
    -------
    data : torch_geometric.data.Data
        The data object containing known training pairs.
    sampling_pool_node_ids : list
        The list of node IDs available for sampling.
    N_pool_pairs : int, optional
        The total number of pairs to return. If None, returns all valid pairs.
    pool_strategy : {'all', 'one_aging', 'both_aging'}, optional
        The strategy for selecting genes (default is 'all').
    min_mutual_interactors : int, optional
        The minimum number of shared neighbors required for a pair to be included.

    Returns:
    -------
    unlabeled_pairs : list of tuples
        A list of (gene_A, gene_B) tuples not present in the training set.
    """
    # Extract known gene pairs to avoid duplicates from training
    known_pairs = set()
    for gA, _, gB, _ in data.pair_pert_group_index.T.cpu().tolist():
        known_pairs.add(tuple(sorted((gA, gB))))

    # Identify lifespan-associated genes globally
    lifespan_set = set()
    if pool_strategy in ['one_aging', 'both_aging']:
        lifespan_indices = data.lifespan_association.cpu().nonzero(as_tuple=True)[0]
        lifespan_set = set(data.node_index[lifespan_indices].cpu().tolist())

    # Generate the full iterator based on strategy using all available nodes
    if pool_strategy == 'all':
        pair_iterator = combinations(sampling_pool_node_ids, 2)

    elif pool_strategy == 'one_aging':
        all_lifespan = [n for n in sampling_pool_node_ids if n in lifespan_set]
        all_non_lifespan = [n for n in sampling_pool_node_ids if n not in lifespan_set]
        # Chain the two valid aging-related pair types
        pair_iterator = chain(product(all_lifespan, all_non_lifespan),
                              combinations(all_lifespan, 2))

    elif pool_strategy == 'both_aging':
        all_lifespan = [n for n in sampling_pool_node_ids if n in lifespan_set]
        pair_iterator = combinations(all_lifespan, 2)

    # Build adjacency only if needed for filtering
    graph_adj = {}
    if min_mutual_interactors > 0:
        edges = data.edge_index.cpu().numpy()
        for u, v in zip(edges[0], edges[1]):
            graph_adj.setdefault(u, set()).add(v)
            graph_adj.setdefault(v, set()).add(u)

    unlabeled_pairs = []
    for gA, gB in pair_iterator:
        if gA == gB: continue
        pair = tuple(sorted((gA, gB)))

        # Skip if already known
        if pair in known_pairs:
            continue

        # Filter by mutual interactors
        if min_mutual_interactors > 0:
            neigh_A = graph_adj.get(gA, set())
            neigh_B = graph_adj.get(gB, set())
            if len(neigh_A.intersection(neigh_B)) < min_mutual_interactors:
                continue

        unlabeled_pairs.append(pair)

    # Randomly shuffle the final valid list and truncate to N_pool_pairs
    random.shuffle(unlabeled_pairs)
    if N_pool_pairs is not None:
        unlabeled_pairs = unlabeled_pairs[:N_pool_pairs]

    return unlabeled_pairs


# --- Acquisition Functions (AL & BO) ---

def select_batch_active_learning(acq_stats, pool_tuples, id2node_dict, query_size=10,
                                 metric='bald_score'):
    """
    Selects the batch that is most *informative* for improving the model.
    Uses high-uncertainty (BALD) or high-entropy sampling.

    Params:
    ------
    acq_stats : dict
        Dictionary of scores calculated from MC dropout samples.
    pool_tuples : list of tuples
        The list of (gene_A, gene_B) pairs in the acquisition pool.
    id2node_dict : dict
        Mapping from gene IDs to names.
    query_size : int, optional
        Number of items to select (default is 10).
    metric : {'bald_score', 'predictive_entropy'}, optional
        The metric used for ranking (default is 'bald_score').

    Returns:
    -------
    selected_batch : list of tuples
        The batch of gene pairs selected for labeling.
    """
    print(f"Selecting batch via Active Learning (Metric: {metric})...")

    scores = acq_stats[metric]

    # Handle case where pool is smaller than query size
    if len(scores) == 0:
        return []
    if len(scores) < query_size:
        print(
            f"  Warning: Pool size ({len(scores)}) is less than query size ({query_size}). "
            f"Returning all {len(scores)} items.")
        query_size = len(scores)

    # Get indices of the top N scores (descending order)
    top_n_indices = np.argsort(scores)[-query_size:][::-1]

    # Print detailed ranked list
    print(f"\n--- Top {query_size} AL Candidates (Metric: {metric}) ---")
    print(f"{'Rank':<4} | {metric:<10} | {'Mutant':<45}")
    print("-" * 70)

    for i, idx in enumerate(top_n_indices):
        # Extract gene IDs from the pool tuple (assuming (gA, gB) format)
        gA_id, gB_id = pool_tuples[idx]

        # Look up names in the dictionary
        gene_name = id2node_dict.get(gA_id, f"ID:{gA_id}")
        partner_name = id2node_dict.get(gB_id, f"ID:{gB_id}")

        # Create the specific string format requested
        mutant_str = f"{gene_name}(kd), {partner_name}(ko)"

        # Print formatted row matching the single_gene_predictions style
        print(f"{i + 1:<4} | {scores[idx]:<10.5f} | {mutant_str}")

    # Select the corresponding tuples for the model return
    selected_batch = [pool_tuples[i] for i in top_n_indices]

    return selected_batch


def select_batch_bayesian_optimization(acq_stats, pool_tuples,
                                       id2node_dict,
                                       best_so_far_value,
                                       query_size=10,
                                       strategy='ucb',
                                       kappa=1.96):
    """
    Selects gene pairs to maximize discovery of "hits" (e.g., synergistic interactions).
    Balances exploration (uncertainty) and exploitation (high mean).

    Params:
    ------
    acq_stats : dict
        Acquisition statistics containing mean and standard deviation.
    pool_tuples : list of tuples
        The list of (gene_A, gene_B) pairs in the acquisition pool.
    id2node_dict : dict
        Mapping from gene IDs to names.
    best_so_far_value : float
        The maximum target probability observed in the training data.
    query_size : int, optional
        Number of items to select (default is 10).
    strategy : {'ucb', 'ei'}, optional
        Bayesian optimization strategy (default is 'ucb').
    kappa : float, optional
        Exploration parameter for UCB (default is 1.96).

    Returns:
    -------
    selected_batch : list of tuples
        The batch of gene pairs selected for labeling.
    """

    print(f"Selecting batch via Bayesian Optimization (Strategy: {strategy})...")

    mean = acq_stats["mean_target_prob"]
    std_dev = acq_stats["std_dev_target_prob"]

    if len(mean) == 0:
        return []

    scores = None
    Z = None

    if strategy.lower() == 'ucb':
        # Score = mean + kappa * std_dev
        scores = mean + kappa * std_dev

    elif strategy.lower() == 'ei':
        epsilon = 1e-9
        # Z-score of the improvement
        Z = (mean - best_so_far_value - epsilon) / (std_dev + epsilon)
        # Expected Improvement calculation
        scores = (mean - best_so_far_value - epsilon) * norm.cdf(Z) + \
                 std_dev * norm.pdf(Z)
        scores[std_dev == 0] = 0.0
    else:
        raise ValueError("Unknown BO strategy. Use 'ucb' or 'ei'.")

    if len(scores) < query_size:
        print(f"  Warning: Pool size ({len(scores)}) less than query size. Returning all items.")
        query_size = len(scores)

    if query_size == 0:
        return []

    top_n_indices = np.argsort(scores)[-query_size:][::-1]

    print(f"\n--- Top {query_size} BO Candidates (Strategy: {strategy}) ---")

    # Shared logic for mutant string formatting
    def get_mutant_str(idx):
        gA_id, gB_id = pool_tuples[idx]
        name_A = id2node_dict.get(gA_id, f"ID:{gA_id}")
        name_B = id2node_dict.get(gB_id, f"ID:{gB_id}")
        return f"{name_A}(kd), {name_B}(ko)"

    if strategy.lower() == 'ucb':
        print(f"{'Rank':<4} | {'UCB Score':<10} | {'Mean Prob':<10} | {'Std Dev':<10} | {'Mutant':<45}")
        print("-" * 85)
        for i, idx in enumerate(top_n_indices):
            mutant_str = get_mutant_str(idx)
            print(f"{i + 1:<4} | {scores[idx]:<10.5f} | {mean[idx]:<10.5f} | {std_dev[idx]:<10.5f} | {mutant_str}")

    elif strategy.lower() == 'ei':
        print(f"{'Rank':<4} | {'EI Score':<10} | {'Mean Prob':<10} | {'Std Dev':<10} | {'Z-Score':<10} | {'Mutant':<45}")
        print("-" * 95)
        for i, idx in enumerate(top_n_indices):
            mutant_str = get_mutant_str(idx)
            print(f"{i + 1:<4} | {scores[idx]:<10.5f} | {mean[idx]:<10.5f} | {std_dev[idx]:<10.5f} | {Z[idx]:<10.5f} | {mutant_str}")

    selected_batch = [pool_tuples[i] for i in top_n_indices]

    return selected_batch

# ----------------------------------------

def calculate_acquisition_stats(mc_predictions, target_class=2):
    """
    Calculates information-theoretic metrics from MC Dropout predictions for active learning.

    Params:
    -------
    mc_predictions : torch.Tensor
        The raw predictions tensor of shape (num_samples, n_pairs, n_classes).
    target_class : int, optional
        The index of the class to focus on for Bayesian Optimization (default is 2, synergy).

    Returns:
    -------
    stats : dict
        Dictionary containing BALD scores, predictive entropy, mean target probability,
        and standard deviation.
    """
    # Mean & Variance across all classes
    #    Shapes: [n_pairs, n_classes]
    mean_probs = mc_predictions.mean(dim=0).numpy()
    pred_variance = mc_predictions.var(dim=0).numpy()

    # Extract stats for the specific target "discovery" class
    #    Shapes: [n_pairs]
    mean_target_prob = mean_probs[:, target_class]
    variance_target_prob = pred_variance[:, target_class]

    # Active Learning (Informativeness) Metrics

    # BALD (Bayesian Active Learning by Disagreement)
    # Mutual Information = Total_Entropy - Expected_Data_Entropy

    # Total Entropy (Entropy of the mean prediction)
    # Convert mean_probs to torch tensor for operations
    mean_probs_tensor = torch.from_numpy(mean_probs)
    total_entropy = -torch.sum(mean_probs_tensor *
                               torch.log(mean_probs_tensor + 1e-9), axis=-1).numpy()

    # Expected Data Entropy (Mean of the entropy of each prediction)
    log_probs = torch.log(mc_predictions + 1e-9)
    entropy_per_sample = -torch.sum(mc_predictions * log_probs, dim=-1)
    expected_data_entropy = entropy_per_sample.mean(dim=0).numpy()

    bald_score = total_entropy - expected_data_entropy

    # Predictive Entropy (simpler uncertainty metric)
    # This is just 'total_entropy' from above.
    # High value means the *average* prediction is uncertain (e.g., [0.5, 0.5])
    predictive_entropy = total_entropy

    return {
        "mean_target_prob": mean_target_prob,  # For BO
        "std_dev_target_prob": np.sqrt(variance_target_prob),  # For BO
        "bald_score": bald_score,  # For AL
        "predictive_entropy": predictive_entropy,  # For AL
        "all_mean_probs": mean_probs,  # For inspection
        "all_pred_variance": pred_variance  # For inspection
    }


def run_acquisition_round(model, data, id2node_dict, acquisition_loader, acquisition_pool, acquisition_goal='active_learning',
                          bo_strategy='ucb', al_metric='bald_score', target_class=2, query_size=10, mc_samples=50,
                          bo_kappa=1.96, device='cpu'):
    """
    Executes a full acquisition cycle to select the next batch of experiments for labeling.

    Params:
    -------
    model : torch.nn.Module
        The current SynergyGT model.
    data : torch_geometric.data.Data
        The experimental and network data.
    id2node_dict : dict
        Mapping from gene IDs to human-readable names.
    acquisition_loader : torch.utils.data.DataLoader
        DataLoader for the candidate pool.
    acquisition_pool : list
        The raw gene pairs corresponding to the loader.
    acquisition_goal : {'active_learning', 'bayesian_optimization'}, optional
        The primary goal of selection.
    bo_strategy : {'ucb', 'ei'}, optional
        The Bayesian Optimization strategy to use.
    al_metric : {'bald_score', 'predictive_entropy'}, optional
        The Active Learning metric to use.
    target_class : int, optional
        The class index to optimize (default is 2).
    query_size : int, optional
        The number of pairs to select (default is 10).
    mc_samples : int, optional
        Number of dropout samples (default is 50).
    bo_kappa : float, optional
        The exploration parameter for UCB (default is 1.96).
    device : str, optional
        The computing device.

    Returns:
    -------
    selected_batch : list
        The list of selected gene pairs.
    """

    # Handle case of empty pool_tuples leading to empty loader
    if len(acquisition_loader) == 0:
        print("Acquisition loader is empty. No predictions to make.")
        return []

    # Run MC Dropout to get predictions
    _, mc_preds = predict_with_mc_dropout(
        model,
        acquisition_loader,
        num_samples=mc_samples,
        device=device
    )

    # Calculate all acquisition statistics
    acq_stats = calculate_acquisition_stats(
        mc_preds,
        target_class=target_class
    )

    # Select the batch based on the chosen goal
    if acquisition_goal == 'active_learning':
        selected_batch = select_batch_active_learning(
            acq_stats,
            acquisition_pool,
            id2node_dict,
            query_size=query_size,
            metric=al_metric
        )

    elif acquisition_goal == 'bayesian_optimization':
        # Get the best score from your training labels
        # This assumes data_train.y holds the *true* scores/labels
        # and they are structured to match the target_class.

        # This is a placeholder - you need to get the *actual*
        # score (e.g., 0.98) from your training labels.
        # If your labels are one-hot, use:
        # best_so_far_value = data_train.y[:, target_class].max().item()

        best_so_far_value = data.pair_effect_type_soft_smoothed[:, target_class].max().item()
        print(f"Using best_so_far_value: {best_so_far_value}")

        selected_batch = select_batch_bayesian_optimization(
            acq_stats,
            acquisition_pool,
            id2node_dict,
            best_so_far_value=best_so_far_value,
            query_size=query_size,
            strategy=bo_strategy,
            kappa=bo_kappa
        )

    else:
        raise ValueError("acquisition_goal must be 'active_learning' or 'bayesian_optimization'")

    print("\n--- Selected Batch for Labeling ---")
    if not selected_batch:
        print("No pairs were selected.")

    # Loop to handle both (gA, gB) and (gA, pA, gB, pB)
    for i, exp in enumerate(selected_batch):
        if len(exp) == 4:
            # Handles the old format with perturbations
            gA, pA, gB, pB = exp
            name_a = id2node_dict.get(gA, f"ID:{gA}")
            name_b = id2node_dict.get(gB, f"ID:{gB}")
            print(f"{i + 1}: {name_a}({pA}), {name_b}({pB})")

        elif len(exp) == 2:
            # Handles the new simplified gene-only format
            gA, gB = exp
            name_a = id2node_dict.get(gA, f"ID:{gA}")
            name_b = id2node_dict.get(gB, f"ID:{gB}")
            print(f"{i + 1}: {name_a}({pA}), {name_b}({pB})")
        else:
            print(f"{i + 1}: {exp}")

    # Clean up GPU memory
    del mc_preds, acquisition_loader

    # Handle MPS, CUDA, or CPU cleanup
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()

    return selected_batch


def estimate_map(samples):
    """
    Estimates the Maximum A Posteriori (MAP) via Kernel Density Estimation.

    Params:
    ------
    samples : np.ndarray
        Array of stochastic samples for a specific class probability.

    Returns:
    -------
    map_estimate : float
        The estimated mode of the probability distribution.
    """
    if np.all(samples == samples[0]):
        return samples[0]
    kernel = stats.gaussian_kde(samples)
    x_range = np.linspace(min(samples), max(samples), 500)
    kde_values = kernel(x_range)
    return x_range[np.argmax(kde_values)]


def plot_uncertainty_densities(pair_results, pair_name="Gene Pair"):
    """
    Visualizes the probability density functions (KDE) for the three interaction
    classes based on MC Dropout samples.

    Params:
    ------
    pair_results : list
        The statistical summary for a single gene pair, typically a list of
        three dictionaries (one per class) containing 'raw_samples', 'map',
        'ci_95', and 'mean'.
    pair_name : str, optional
        The human-readable name of the gene pair (e.g., "DAF-16 + AGE-1")
        used for the plot title (default is "Gene Pair").

    Returns:
    -------
    None
    """
    classes = ["Antagonistic", "Additive", "Synergistic"]
    colors = ["#3498db", "#95a5a6", "#e74c3c"] # Blue, Gray, Red

    plt.figure(figsize=(10, 6))

    for i, class_name in enumerate(classes):
        data = pair_results[i]
        samples = data['raw_samples']

        # Plot Kernel Density Estimate
        sns.kdeplot(samples, fill=True, label=f"{class_name}", color=colors[i], alpha=0.3)

        # Draw vertical lines for MAP and CI
        plt.axvline(data['map'], color=colors[i], linestyle='--', lw=2,
                    label=f"{class_name} MAP: {data['map']:.3f}")

        print(f"--- {class_name} Interaction Statistics ---")
        print(f"MAP Estimate: {data['map']:.4f}")
        print(f"95% CI:       [{data['ci_95'][0]:.4f}, {data['ci_95'][1]:.4f}]")
        print(f"Mean (μ):     {data['mean']:.4f}\n")

    plt.title(f"Interaction Type Probability Distributions: {pair_name}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_average_distribution(dataloader, label_name, num_classes, device):
    """
    Computes the average label distribution across a given dataset.

    Params:
    ------
    dataloader : torch.utils.data.DataLoader
        Dataloader containing target labels.
    label_name : str
        The key for the target labels.
    num_classes : int
        The number of classes in the label distribution.
    device : torch.device or str
        The device to run the calculation on.

    Returns:
    -------
    avg_dist : torch.Tensor
        The mean distribution across the dataset.
    """
    total_dist = torch.zeros(num_classes, device=device)
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch[label_name].to(device)
            total_dist += labels.sum(dim=0)
            total_samples += labels.size(0)
    return (total_dist / total_samples).cpu()


def naive_baseline(model, train_loader, test_loader, loss_fn, baseline_type='mean', smoothed_label=True,
                      C=3, device=None):
    """
    Initializes and evaluates a naive baseline model (Mean or Uniform).

    Params:
    ------
    model : torch.nn.Module
        A baseline model instance.
    train_loader : torch.utils.data.DataLoader
        Dataloader used to calculate mean distribution (for 'mean' baseline).
    test_loader : torch.utils.data.DataLoader
        Dataloader for model evaluation.
    loss_fn : callable
        Loss function for evaluation.
    baseline_type : {'mean', 'uniform'}, optional
        The type of naive strategy to employ (default is 'mean').
    smoothed_label : bool, optional
        Whether to use smoothed target labels (default is True).
    C : int, optional
        Number of interaction classes (default is 3).
    device : torch.device, optional
        Device to run evaluation on.

    Returns:
    -------
    test_results : dict
        Performance metrics for the naive baseline.
    """
    if baseline_type not in ['mean', 'uniform']:
        raise ValueError("baseline_type must be 'mean' or 'uniform'")

    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    label_name = "target_soft_smoothed" if smoothed_label else "target_soft"

    if baseline_type == 'mean':
        print("Calculating average distribution on the training data for this fold...")
        avg_dist = calculate_average_distribution(train_loader, label_name, C, device)

        print("Initializing the naive model with the calculated average...")
        model = NaiveBaseline(avg_dist).to(device)

    elif baseline_type == 'uniform':
        print("Setting up the uniform distribution baseline...")

        print("Initializing the uniform baseline model...")
        model = UniformBaseline(C).to(device)

    # --- Evaluation Step ---
    print("Evaluating the model on the held-out data...")

    # Define the torchmetrics for evaluation
    test_metrics_soft = {
        "auc": AUROC(task="multiclass", num_classes=C, average="macro").to(device),
        "auc_per_class": AUROC(task="multiclass", num_classes=C, average="none").to(device),
        "auprc": AveragePrecision(task="multiclass", num_classes=C, average="macro").to(device),
        "auprc_per_class": AveragePrecision(task="multiclass", num_classes=C, average="none").to(device)
    }

    test_loss, test_metrics = eval_epoch_trans(
        model=model,
        dataloader=test_loader,
        label_name=label_name,
        criterion=loss_fn,
        device=device,
        metrics=test_metrics_soft
    )

    test_results = {
        name: m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m
        for name, m in test_metrics.items()
    }
    test_results["loss_kl"] = test_loss

    # Clean up memory
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return test_results


def _get_representations(model, dataloader, device):
    """
    Extracts the model's internal latent representations for nodes or pairs.

    Params:
    ------
    model : torch.nn.Module
        The model used for feature extraction.
    dataloader : torch.utils.data.DataLoader
        The data to pass through the model.
    device : str or torch.device
        The computing device.

    Returns:
    -------
    reps : np.ndarray
        The extracted latent features.
    labels : np.ndarray
        The corresponding ground-truth soft labels.
    """
    model.eval()
    model.to(device)
    reps, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            _, pair_rep = model(batch, return_representation=True)
            reps.append(pair_rep.cpu().numpy())
            labels.append(batch['target_soft_smoothed'].cpu().numpy())

    return np.concatenate(reps, axis=0), np.concatenate(labels, axis=0)


def visualize_synergy_landscape(model, train_loader, test_loader, method='tsne', device='mps'):
    """
    Generates a 2D projection (t-SNE or UMAP) of the model's internal gene-pair representations.

    Params:
    -------
    model : torch.nn.Module
        The trained SynergyGT model.
    train_loader : torch.utils.data.DataLoader
        The training data to project.
    test_loader : torch.utils.data.DataLoader
        The testing data to project.
    method : {'tsne', 'umap'}, optional
        The dimensionality reduction technique (default is 'tsne').
    device : str, optional
        The computing device.
    """
    # Extract data
    print("Extracting [PAIR] representations...")
    X_train, y_train = _get_representations(model, train_loader, device)
    X_test, y_test = _get_representations(model, test_loader, device)

    # Combine for a joint dimensionality reduction (ensures same space)
    X_combined = np.concatenate([X_train, X_test], axis=0)
    n_train = X_train.shape[0]

    # Run Dimensionality Reduction
    if method.lower() == 'tsne':
        print(f"Running t-SNE on {X_combined.shape[0]} total pairs... ({X_train.shape[0]} train, {X_test.shape[0]} test)")
        reducer = TSNE(n_components=2, perplexity=30, random_state=23, init='pca', learning_rate='auto')
        X_emb = reducer.fit_transform(X_combined)
        label_prefix = "t-SNE"
    elif method.lower() == 'umap':
        print(f"Running UMAP on {X_combined.shape[0]} total pairs... ({X_train.shape[0]} train, {X_test.shape[0]} test)")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=23)
        X_emb = reducer.fit_transform(X_combined)
        label_prefix = "UMAP"
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    X_train_emb = X_emb[:n_train]
    X_test_emb = X_emb[n_train:]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    datasets = [
        (X_train_emb, y_train, "Training Set", axes[0]),
        (X_test_emb, y_test, "Testing Set", axes[1])
    ]

    for X_plot, y_raw, title, ax in datasets:
        df = pd.DataFrame({'x': X_plot[:, 0], 'y': X_plot[:, 1]})
        y_indices = np.argmax(y_raw, axis=1)
        df['Class'] = ['Synergistic' if idx == 2 else 'Not Synergistic' for idx in y_indices]
        df['Frequency'] = y_raw[:, 2]
        df = df.sort_values(by='Frequency')  # Plot synergistic on top

        sns.scatterplot(
            data=df, x='x', y='y', hue='Class', ax=ax,
            palette={'Synergistic': 'red', 'Not Synergistic': '#eeeeee'},
            alpha=0.7, edgecolor='none', s=15, legend=True
        )

        ax.set_title(f"[PAIR] Representation ({label_prefix}) \n {title}", fontsize=14)
        ax.set_xlabel(f"{label_prefix} 1")
        ax.set_ylabel(f"{label_prefix} 2")
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()


def analyze_top_attended_nodes(model, loader, id_to_name_map, device, top_k=20, min_occurrences=0):
    """
    Analyzes Transformer attention weights to identify which nodes the model focuses on most.

    Params:
    -------
    model : torch.nn.Module
        The SynergyGT model with attention recording enabled.
    loader : torch.utils.data.DataLoader
        The data to analyze.
    id_to_name_map : dict
        Mapping from IDs to gene names.
    device : str
        The computing device.
    top_k : int, optional
        Number of top nodes to display in the summary table (default is 20).
    min_occurrences : int, optional
        Minimum times a node must appear in the data to be considered (default is 0).
    """
    model.to(device)
    model.eval()

    pair_attn_sum = defaultdict(float)
    all_tokens_attn_sum = defaultdict(float)
    node_occurrence_count = defaultdict(int)

    # Metadata Caches
    node_lifespan_status = {}
    node_in_degree = {}
    node_out_degree = {}

    print(f"--- Starting Top-{top_k} Node Attention Analysis ---")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scanning Node Attention"):

            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v

            logits, batch_attn_weights = model(batch_on_device, return_attention=True)

            # Last layer attention: (B, Heads, Seq, Seq) -> Average heads: (B, Seq, Seq)
            avg_attn = batch_attn_weights[-1].mean(dim=1)

            # Slice off [PAIR] token to align with N nodes
            pair_to_nodes_attn = avg_attn[:, 0, 1:]
            avg_incoming = avg_attn.mean(dim=1)
            all_to_nodes_attn = avg_incoming[:, 1:]

            # Metadata extraction
            if 'node_ids' in batch_on_device:
                batch_node_ids = batch_on_device['node_ids'].cpu().numpy()
            elif 'x' in batch_on_device:
                batch_node_ids = batch_on_device['x'].squeeze().cpu().numpy()
            else:
                continue

            # Extract lifespan info
            if 'is_lifespan_gene' in batch_on_device:
                batch_is_lifespan = batch_on_device['is_lifespan_gene'].cpu().numpy()
            else:
                batch_is_lifespan = np.zeros_like(batch_node_ids, dtype=bool)

            # Extract Centrality (Degree) info
            if 'in_degree' in batch_on_device:
                batch_in_degree = batch_on_device['in_degree'].cpu().numpy()
            else:
                batch_in_degree = np.zeros_like(batch_node_ids, dtype=int)

            if 'out_degree' in batch_on_device:
                batch_out_degree = batch_on_device['out_degree'].cpu().numpy()
            else:
                batch_out_degree = np.zeros_like(batch_node_ids, dtype=int)

            dist_to_u = batch_on_device['dist_to_u'].cpu()
            dist_to_v = batch_on_device['dist_to_v'].cpu()
            padding_mask = batch_on_device['padding_mask'].cpu().bool()

            batch_size = pair_to_nodes_attn.shape[0]
            pair_to_nodes_attn = pair_to_nodes_attn.cpu()
            all_to_nodes_attn = all_to_nodes_attn.cpu()

            for i in range(batch_size):
                is_focal = (dist_to_u[i] == 0) | (dist_to_v[i] == 0)
                is_pad = padding_mask[i]
                valid_mask = (~is_focal) & (~is_pad)

                # Handle batch dimensions for metadata
                if batch_node_ids.ndim > 1:
                    current_ids = batch_node_ids[i]
                    current_lifespan = batch_is_lifespan[i]
                    current_in_deg = batch_in_degree[i]
                    current_out_deg = batch_out_degree[i]
                else:
                    current_ids = batch_node_ids[i]
                    current_lifespan = batch_is_lifespan[i]
                    current_in_deg = batch_in_degree[i]
                    current_out_deg = batch_out_degree[i]

                # Safety check for size mismatches
                if len(valid_mask) != pair_to_nodes_attn.shape[1]:
                    min_len = min(len(valid_mask), pair_to_nodes_attn.shape[1])
                    valid_mask = valid_mask[:min_len]
                    p_scores_i = pair_to_nodes_attn[i][:min_len]
                    a_scores_i = all_to_nodes_attn[i][:min_len]
                    current_ids = current_ids[:min_len]
                    current_lifespan = current_lifespan[:min_len]
                    current_in_deg = current_in_deg[:min_len]
                    current_out_deg = current_out_deg[:min_len]
                else:
                    p_scores_i = pair_to_nodes_attn[i]
                    a_scores_i = all_to_nodes_attn[i]

                valid_indices = torch.nonzero(valid_mask).squeeze()
                if valid_indices.numel() == 0:
                    continue

                p_scores = p_scores_i[valid_mask]
                a_scores = a_scores_i[valid_mask]
                target_ids = current_ids[valid_mask]
                target_lifespan = current_lifespan[valid_mask]
                target_in_deg = current_in_deg[valid_mask]
                target_out_deg = current_out_deg[valid_mask]

                for node_id, p_score, a_score, is_life, in_d, out_d in zip(target_ids, p_scores, a_scores,
                                                                           target_lifespan, target_in_deg,
                                                                           target_out_deg):
                    nid = int(node_id) if hasattr(node_id, 'item') else node_id

                    pair_attn_sum[nid] += p_score.item()
                    all_tokens_attn_sum[nid] += a_score.item()
                    node_occurrence_count[nid] += 1

                    # Cache static properties (Idempotent updates)
                    if nid not in node_lifespan_status:
                        node_lifespan_status[nid] = bool(is_life)
                    if nid not in node_in_degree:
                        node_in_degree[nid] = int(in_d)
                    if nid not in node_out_degree:
                        node_out_degree[nid] = int(out_d)

    final_pair_stats = []
    final_all_stats = []

    for nid, count in node_occurrence_count.items():
        if count < min_occurrences:
            continue

        avg_pair = pair_attn_sum[nid] / count
        avg_all = all_tokens_attn_sum[nid] / count
        is_lifespan = node_lifespan_status.get(nid, False)
        in_d = node_in_degree.get(nid, 0)
        out_d = node_out_degree.get(nid, 0)

        name = str(nid)
        if id_to_name_map and nid in id_to_name_map:
            str_name = id_to_name_map[nid]
            name = f"{str_name} ({nid})"

        # Store stats tuples
        final_pair_stats.append((name, avg_pair, count, is_lifespan, in_d, out_d))
        final_all_stats.append((name, avg_all, count, is_lifespan, in_d, out_d))

    final_pair_stats.sort(key=lambda x: x[1], reverse=True)
    final_all_stats.sort(key=lambda x: x[1], reverse=True)

    def print_table(title, data):
        base_names = []
        print(f"\n=== {title} (Top {top_k}) ===")
        # Header with new columns
        print(
            f"{'Rank':<5} | {'Node Name/ID':<30} | {'Avg Attn':<12} | {'Occur':<6} | {'Aging':<6} | {'In-Deg':<6} | {'Out-Deg':<6}")
        print("-" * 100)
        for rank, (name, score, count, is_ls, in_d, out_d) in enumerate(data[:top_k], 1):
            ls_str = "YES" if is_ls else "No"
            print(f"{rank:<5} | {name:<30} | {score:.6f}     | {count:<6} | {ls_str:<6} | {in_d:<6} | {out_d:<6}")
            base_name = name.split(' ')[0]
            base_names.append(base_name)
        print(f"Gene names: {', '.join(base_names)}")

    print_table("Highest Attention from [PAIR] Token", final_pair_stats)
    print_table("Highest Attention from ALL Tokens (Avg Incoming)", final_all_stats)

    print("\n=== Dataset Composition Stats ===")
    total_unique_nodes = len(node_lifespan_status)
    total_lifespan_nodes = sum(node_lifespan_status.values())

    if total_unique_nodes > 0:
        lifespan_fraction = total_lifespan_nodes / total_unique_nodes
        print(f"Total Unique Nodes Analyzed: {total_unique_nodes}")
        print(f"Total Lifespan Genes:        {total_lifespan_nodes}")
        print(f"Fraction of Lifespan Genes:  {lifespan_fraction:.4f} ({lifespan_fraction * 100:.2f}%)")
    else:
        print("No valid nodes encountered to calculate stats.")
    print("=================================\n")


def plot_subgraph_sample(subgraph_data, data_network, id2node_dict):
    """
    Visualizes a single subgraph sample with focal genes highlighted.

    Params:
    ------
    subgraph_data : list
        List containing a single subgraph dictionary.
    data_network : torch_geometric.data.Data
        The global network object.
    id2node_dict : dict
        Mapping from gene IDs to names.

    Returns:
    -------
    None
    """
    if len(subgraph_data) != 1:
        raise ValueError(f"Expected 1 subgraph, but got {len(subgraph_data)}")

    sample = subgraph_data[0]
    subset_nodes = sample['node_ids']
    focal_pair = sample['pair']

    edge_index_sub, _ = subgraph(
        subset_nodes,
        data_network.edge_index,
        relabel_nodes=False
    )

    G = nx.DiGraph()
    G.add_nodes_from(subset_nodes.tolist())
    G.add_edges_from(edge_index_sub.t().tolist())

    node_colors = []
    for node in G.nodes():
        if node in focal_pair.tolist():
            node_colors.append('#ff6b6b')
        else:
            node_colors.append('#4ecdc4')

    node_labels = {node: id2node_dict.get(node, node) for node in G.nodes()}
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True, arrowstyle='-|>', arrowsize=20,
                           node_size=800)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_weight='bold')
    name_a = id2node_dict[focal_pair[0].item()]
    name_b = id2node_dict[focal_pair[1].item()]

    focal_pair_label = f"({name_a}, {name_b})"
    plt.title(f"Subgraph of {focal_pair_label}")
    plt.axis('off')
    plt.show()


def analyze_subgraph_sample(subgraph_data, id2node_dict):
    """
    Prints descriptive statistics for a specific subgraph (distances and edge types).

    Params:
    ------
    subgraph_data : list
        List containing a single subgraph dictionary.
    id2node_dict : dict
        Mapping from gene IDs to names.

    Returns:
    -------
    None
    """

    if len(subgraph_data) != 1:
        raise ValueError(f"Expected 1 subgraph, but got {len(subgraph_data)}")

    sample = subgraph_data[0]

    # Extract global IDs and perturbations for the pair
    u_global, v_global = sample['pair'][0].item(), sample['pair'][1].item()
    name_u = id2node_dict[u_global]
    name_v = id2node_dict[v_global]
    u_perturb, v_perturb = sample['pair_perturbations'][0].item(), sample['pair_perturbations'][1].item()

    # Find local indices in the node_ids tensor
    node_ids = sample['node_ids']
    idx_u = (node_ids == u_global).nonzero(as_tuple=True)[0].item()
    idx_v = (node_ids == v_global).nonzero(as_tuple=True)[0].item()

    # Distances
    hop_dist = sample['pairwise_dist'][idx_u, idx_v].item()
    aging_dist_u = sample['lifespan_dist'][idx_u].item()
    aging_dist_v = sample['lifespan_dist'][idx_v].item()

    print(f"=== Subgraph Summary ({name_u} & {name_v}) ===")
    print(f"\nHop distance between nodes: {hop_dist}   (1 => nodes interact directly)")
    print(f"\nAging distance (lifespan_dist) - Gene U: {aging_dist_u}, Gene V: {aging_dist_v}   (0 => is a known lifespan"
          f" gene, 1 => interacts with a known lifespan gene)")

    # Edge type frequency summary
    # 0: genetic, 1: regulatory, 2: physical, 3: none
    adj = sample['adj_matrix']

    # Filter out the '3' (no edge) values
    edges_only = adj[adj != 3]

    # Count occurrences of 0, 1, and 2
    counts = torch.bincount(edges_only.flatten(), minlength=3)

    edge_map = {0: "Genetic", 1: "Regulatory", 2: "Physical"}
    total_edges = edges_only.numel()

    print(f"\n=== Subgraph Edge Summary ===")
    print(f"Total interactions/edges: {total_edges}")
    for val, name in edge_map.items():
        count = counts[val].item()
        percentage = (count / total_edges * 100) if total_edges > 0 else 0
        print(f"- {name} (Type {val}): {count} ({percentage:.1f}%)")


def single_gene_summary(gene_id, data_original, id2node_dict):
    """
    Provides a biological and network-centric summary for a specific gene in the network.

    Params:
    -------
    gene_id : int
        The internal node ID of the gene.
    data_original : torch_geometric.data.Data
        The full graph and experimental dataset.
    id2node_dict : dict
        Mapping from IDs to gene names.
    """
    # Mapping for perturbation IDs based on typical project schema:
    # 0=knockdown, 1=knockout, 2=overexpression
    pert_map = {0: "kd", 1: "ko", 2: "oe", 3: "pad"}

    gene_name = id2node_dict.get(gene_id, f"Unknown({gene_id})")
    print(f"=== Summary for Gene: {gene_name} (ID: {gene_id}) ===")

    # Centrality (in-degree, out-degree)
    out_deg = degree(data_original.edge_index[0], num_nodes=data_original.num_nodes)
    in_deg = degree(data_original.edge_index[1], num_nodes=data_original.num_nodes)
    g_in = in_deg[gene_id].item()
    g_out = out_deg[gene_id].item()
    print(f"- Centrality: In-Degree = {g_in}, Out-Degree = {g_out}")

    # Interaction type breakdown
    mask_source = data_original.edge_index[0] == gene_id
    mask_target = data_original.edge_index[1] == gene_id
    relevant_edges_mask = mask_source | mask_target
    gene_edge_types = data_original.edge_type[relevant_edges_mask]
    total_interactions = gene_edge_types.numel()

    if total_interactions > 0:
        counts = torch.bincount(gene_edge_types, minlength=3)
        perc = (counts.float() / total_interactions) * 100
        print(f"- Interaction Types (Total: {total_interactions}):")
        print(f"  * Genetic:    {counts[0]} ({perc[0]:.1f}%)")
        print(f"  * Regulatory: {counts[1]} ({perc[1]:.1f}%)")
        print(f"  * Physical:   {counts[2]} ({perc[2]:.1f}%)")
    else:
        print("- Interaction Types: No interactions found in graph.")

    # Lifespan association
    is_lifespan = data_original.lifespan_association[gene_id].item()
    ls_status = "YES" if is_lifespan == 1 else "No"
    print(f"- Known Lifespan Association: {ls_status}")

    # Neighbors lifespan proportion
    neighbors_out = data_original.edge_index[1][mask_source]
    neighbors_in = data_original.edge_index[0][mask_target]
    all_neighbors = torch.unique(torch.cat([neighbors_out, neighbors_in]))

    if all_neighbors.numel() > 0:
        neighbor_ls_flags = data_original.lifespan_association[all_neighbors]
        prop_ls_neighbors = (neighbor_ls_flags.sum().float() / all_neighbors.numel()) * 100
        # print(f"- Neighbors: {all_neighbors.numel()} unique neighbors, {prop_ls_neighbors:.1f}% are lifespan genes")
        print(f"- Neighbors: {all_neighbors.numel()} unique neighbors, {neighbor_ls_flags.sum()} are lifespan genes")

    else:
        print("- Neighbors: No neighbors found.")

    # Experimental data availability
    exp_mask = (data_original.pair_pert_group_index[0] == gene_id) | \
               (data_original.pair_pert_group_index[2] == gene_id)

    in_experimental = exp_mask.any().item()
    print(f"- Appears in Experimental Data: {'Yes' if in_experimental else 'No'}")

    # Experimental data breakdown
    if in_experimental:
        indices = torch.where(exp_mask)[0]
        avg_dist = data_original.pair_effect_type_soft_smoothed[exp_mask].mean(dim=0)

        print(f"- Experimental Details:")
        print(f"  * Involved in {len(indices)} unique tested pairs")
        print(f"  * Avg. Interaction Distribution [Antag / Neither / Synerg]:")
        print(f"    [{avg_dist[0]:.3f} / {avg_dist[1]:.3f} / {avg_dist[2]:.3f}]")

        print(f"\n  * Full Experiment List (w/ smoothed interaction type frequencies):")
        print(f"    {'Mutant':<40} | {'Distribution [A/N/S]':<20}")
        print(f"    {'-' * 65}")

        for idx in indices:
            u_id = data_original.pair_pert_group_index[0, idx].item()
            u_pert = data_original.pair_pert_group_index[1, idx].item()
            v_id = data_original.pair_pert_group_index[2, idx].item()
            v_pert = data_original.pair_pert_group_index[3, idx].item()

            u_name = id2node_dict.get(u_id, f"ID:{u_id}")
            v_name = id2node_dict.get(v_id, f"ID:{v_id}")
            u_p_name = pert_map.get(u_pert, str(u_pert))
            v_p_name = pert_map.get(v_pert, str(v_pert))

            exp_str = f"{u_name}({u_p_name}) + {v_name}({v_p_name})"

            dist = data_original.pair_effect_type_soft_smoothed[idx].tolist()
            print(f"    {exp_str:<40} | [{dist[0]:.3f}, {dist[1]:.3f}, {dist[2]:.3f}]")

    print("=" * 70)


def single_gene_predictions(predictions, id2node_dict, gene_id, top_n=10):
    """
    Aggregates and ranks model predictions involving a specific gene of interest.

    Params:
    ------
    predictions : dict
        The output from get_predictions_synergy_model.
    id2node_dict : dict
        Mapping from gene IDs to names.
    gene_id : int
        The target gene's node ID.
    top_n : int, optional
        Number of top synergistic partners to display (default is 10).

    Returns:
    -------
    None
    """
    pairs = predictions['pairs']
    preds = np.array(predictions['preds'])
    gene_name = id2node_dict.get(gene_id, f"ID:{gene_id}")

    # 1. Filter for the target gene
    mask = [(gene_id in p) for p in pairs]
    filtered_pairs = [p for i, p in enumerate(pairs) if mask[i]]
    filtered_preds = preds[mask]

    if len(filtered_preds) == 0:
        print(f"No predictions found for gene {gene_name} (ID: {gene_id}).")
        return None

    # 2. Extract partner IDs
    partners = [p[0] if p[1] == gene_id else p[1] for p in filtered_pairs]
    avg_dist = np.mean(filtered_preds, axis=0)

    # 3. Sort by Synergy Likelihood (Index 2)
    ranked_indices = np.argsort(filtered_preds[:, 2])[::-1]

    # 4. Print formatted output
    print(f"=== Model Predictions for Gene: {gene_name} (ID: {gene_id}) ===")
    print(f"- Evaluation Details:")
    print(f"  * Total Pairs Evaluated: {len(filtered_preds)}")
    print(f"  * Avg. Predicted Distribution [Antag / Addit / Synerg]:")
    print(f"    [{avg_dist[0]:.3f} / {avg_dist[1]:.3f} / {avg_dist[2]:.3f}]")

    print(f"\n  * Top {top_n} most synergistic double mutants:")
    print(f"    {'Mutant':<45} | {'Synergistic Prob.':<20}")
    print(f"    {'-' * 70}")

    ranking_list = []

    for i in ranked_indices[:top_n]:
        p_id = partners[i]
        partner_name = id2node_dict.get(p_id, f"ID:{p_id}")
        synergy_prob = filtered_preds[i, 2]

        # Format the mutant column as requested
        mutant_str = f"{gene_name}(kd), {partner_name}(ko)"

        # Print formatted row
        print(f"    {mutant_str:<45} | {synergy_prob:.3f}")

        ranking_list.append({
            'mutant': mutant_str,
            'synergistic_prob': synergy_prob
        })

    print("=" * 75)

    # return {
    #     "avg_dist": avg_dist,
    #     "ranking": pd.DataFrame(ranking_list)
    # }
