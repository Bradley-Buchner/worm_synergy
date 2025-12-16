import torch.nn.functional as F
from torchmetrics import AUROC, AveragePrecision
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict

from model.eval import *
from model.models import NaiveBaseline, UniformBaseline


def train_test_split_simple(processed_data, train_frac=0.8, seed=23):
    """
    Splits data into train/test sets by shuffling.
    This method assumes each item in processed_data corresponds to a unique pair.
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
    Evaluates the model and calculates metrics, including macro-average
    and per-class AUC scores.
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
    Evaluates the model and calculates metrics, including macro-average
    and per-class AUC scores. If plot_fig=True, generates a scatterplot of
    Predicted vs Actual Synergy Probabilities.
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1:03d}")
        train_loss_epoch = train_epoch_trans(model, train_loader, label_name, optimizer, loss_fn, device,
                                             randomize_labels=randomize_labels)

        test_loss_epoch, *_ = eval_epoch_trans_plt(model, test_loader, label_name, loss_fn, device,
                                                  metrics={}, plot_fig=False)

        model.history["train_loss"].append(train_loss_epoch)
        model.history["test_loss"].append(test_loss_epoch)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"lr={current_lr}  train_loss={train_loss_epoch:.4f}  "
              f"test_loss={test_loss_epoch:.4f}\n")

    print(f"Training complete!")

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

    return results, test_preds_final


def get_predictions_synergy_model(model, dataloader, label_name, device):
    """
    Evaluates the model and returns predictions.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            labels = batch.pop(label_name).to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(inputs)
            preds_probs = F.softmax(outputs, dim=1)

            all_preds.extend(preds_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    predictions = {"preds": all_preds, "labels": all_labels}

    return predictions


def get_synergy_model_performance(model_metrics, uninformative_baseline_metrics, mean_baseline_metrics):
    import pandas as pd

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


def calculate_average_distribution(dataloader, label_name, num_classes, device):
    """
    Calculates the mean label distribution across a dataset.
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
    Evaluates a naive baseline model on a dataset using the specified baseline type,
    either baseline_type = 'mean' or 'uniform'.
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


def visualize_synergy_landscape(model, dataloader, color_by='class', device='mps'):
    model.eval()
    model.to(device)

    representations = []
    labels = []

    print("Extracting pair representations...")
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            _, pair_rep = model(batch, return_representation=True)
            representations.append(pair_rep.cpu().numpy())

            labels.append(batch['target_soft_smoothed'].cpu().numpy())

    X = np.concatenate(representations, axis=0)

    print(f"Running t-SNE on {X.shape[0]} pairs...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=23, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))

    # Check if we have labels to plot
    if len(labels) > 0:
        y_raw = np.concatenate(labels, axis=0)
        df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1]})

        df['Synergy Frequency'] = y_raw[:, 2]

        if color_by == 'class':
            y_indices = np.argmax(y_raw, axis=1)
            classes = ['Synergistic' if idx == 2 else 'Not Synergistic' for idx in y_indices]
            df['Class'] = classes

            # Sort by frequency so larger/synergistic dots plot on top of smaller ones
            df = df.sort_values(by='Synergy Frequency')

            sns.scatterplot(
                data=df, x='x', y='y',
                hue='Class',               # Color determined by Class
                # size='Synergy Frequency',      # Size determined by Frequency
                # sizes=(10, 200),                # Range: Small dots (low freq) -> Big dots (high freq)
                palette={'Synergistic': 'red', 'Not Synergistic': '#eeeeee'}, # Red vs Dark Grey
                alpha=0.85,
                edgecolor='none'
            )
            plt.title("t-SNE of [PAIR] Token Representation (Training Set)")

    else:
        # Fallback if no labels exist
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.5, s=20, c='steelblue')
        plt.title("Synergy Landscape (Unlabeled)")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Visual cleanup
    sns.despine()
    plt.grid(False) # Clean look
    plt.show()


def analyze_top_attended_nodes(model, loader, id_to_name_map, device, top_k=20, min_occurrences=0):
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