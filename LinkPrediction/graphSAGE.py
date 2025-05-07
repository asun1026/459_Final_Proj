#!/usr/bin/env python
# -*- coding: utf-8 -*-
# graphSAGE.py

import os
import json
import time
import numpy as np
import networkx as nx
import torch
from torch import Tensor
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv

from sklearn.metrics import roc_auc_score, precision_score
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
# (Keep your existing configurations for paths, EMBEDDING_FILES, LEARNABLE_EMB_DIM, GNN Hyperparams, etc.)
# Data Paths
DATA_EXTRACT_DIR = "DataExtract/Data"
LLM_EMBED_DIR = "LLM_embed/embed_data"
GRAPH_FILE = os.path.join(DATA_EXTRACT_DIR, "stackexchange_graph.gexf")
MAPPING_FILE = os.path.join(DATA_EXTRACT_DIR, "url_id_mapping.json")

LEARNABLE_EMB_DIM = 128
EMBEDDING_FILES = {
    "LearnableEmb": None,
    "SBERT": os.path.join(LLM_EMBED_DIR, "node_sbert_embeddings.npz"),
    "Layer_04": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_04.npz"),
    "Layer_08": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_08.npz"),
    "Layer_12": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_12.npz"),
    "Layer_16": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_16.npz"),
    "Layer_20": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_20.npz"),
    "Layer_24": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_24.npz"),
    "Layer_28": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_28.npz"),
    "Layer_31": os.path.join(LLM_EMBED_DIR, "node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_32.npz"), 
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0
EPOCHS = 50 # Adjust as needed
VAL_EVAL_FREQ = 5 # How often to evaluate on validation set during training
BATCH_SIZE = 2048
HIDDEN_DIM = 128
OUTPUT_DIM = 64
NUM_GNN_LAYERS = 2
DROPOUT = 0.3
LINK_PRED_LAYERS = 3
THRESHOLD_FOR_PRECISION = 0.5
K_VALS_HITS = [10, 50, 100]

# --- Setup Logging ---
log_file_handler = logging.FileHandler("gnn_multi_embed_comparison.log", mode='w')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[log_file_handler, logging.StreamHandler()]
)
logging.info(f"Using device: {DEVICE}")

# --- GNN Model & Link Predictor Definitions --- (Keep as is)
class GNNStack(torch.nn.Module): # ... (Your GNNStack definition) ...
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
        super(GNNStack, self).__init__()
        conv_model = SAGEConv
        self.convs = nn.ModuleList()
        if num_layers == 1: self.convs.append(conv_model(input_dim, output_dim))
        else:
            self.convs.append(conv_model(input_dim, hidden_dim))
            for _ in range(num_layers - 2): self.convs.append(conv_model(hidden_dim, hidden_dim))
            self.convs.append(conv_model(hidden_dim, output_dim))
        self.dropout_rate = dropout; self.num_layers = num_layers; self.emb = emb
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1: x = F.relu(x); x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x if self.emb else F.log_softmax(x, dim=1)

class LinkPredictor(nn.Module): # ... (Your LinkPredictor definition) ...
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1: self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2): self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout_rate = dropout
    def reset_parameters(self):
        for lin in self.lins: lin.reset_parameters()
    def forward(self, x_i, x_j):
        x = x_i * x_j
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1: x = F.relu(x); x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return torch.sigmoid(x)

# --- Hard Negative Sampling Helpers --- (Keep as is)
def get_two_hop_non_neighbors(node_u: int, num_nodes: int, adj_matrix: Tensor, all_true_edges_set: set) -> list: # ...
    if node_u >= adj_matrix.size(0): return []
    one_hop_u_mask = adj_matrix[node_u] > 0
    u_neighbors_indices = one_hop_u_mask.nonzero(as_tuple=False).view(-1)
    potential_two_hop_nodes = set()
    for neighbor_idx in u_neighbors_indices:
        if neighbor_idx.item() >= adj_matrix.size(0): continue
        neighbors_of_neighbor = (adj_matrix[neighbor_idx.item()] > 0).nonzero(as_tuple=False).view(-1)
        for two_hop_node_idx in neighbors_of_neighbor: potential_two_hop_nodes.add(two_hop_node_idx.item())
    hard_negative_candidates = []
    for w_candidate in potential_two_hop_nodes:
        if w_candidate != node_u and \
           (w_candidate >= num_nodes or not one_hop_u_mask[w_candidate]) and \
           tuple(sorted((node_u, w_candidate))) not in all_true_edges_set:
            hard_negative_candidates.append(w_candidate)
    return hard_negative_candidates

def generate_hard_negatives_by_common_neighbors( # Renamed from generate_hard_negative_set for clarity
    positive_edges: Tensor, num_nodes: int,
    all_true_edges_set: set, adj_matrix: Tensor,
    num_candidates_to_rank: int = 50, num_neg_per_pos_edge: int = 1
) -> Tensor: # ... (Your chosen hard negative sampler, e.g., common neighbors) ...
    hard_neg_sources, hard_neg_targets = [], []
    if positive_edges.size(1) == 0: return torch.empty(2, 0, dtype=torch.long, device=positive_edges.device)
    
    for i in range(positive_edges.size(1)):
        u = positive_edges[0, i].item(); v_pos = positive_edges[1, i].item() 
        found_neg_for_this_pos_edge = False; best_w_hard = -1
        current_candidates = [] 
        considered_w_for_ranking = set()
        for _ in range(num_candidates_to_rank):
            w_cand = random.randint(0, num_nodes - 1)
            if w_cand == u or w_cand == v_pos or tuple(sorted((u, w_cand))) in all_true_edges_set or w_cand in considered_w_for_ranking:
                continue
            considered_w_for_ranking.add(w_cand)
            cn_count = count_common_neighbors(u, w_cand, adj_matrix) # Assumes count_common_neighbors is defined
            current_candidates.append({'w': w_cand, 'cn_count': cn_count})
        if current_candidates:
            current_candidates.sort(key=lambda x: x['cn_count'], reverse=True)
            best_w_hard = current_candidates[0]['w']
        if best_w_hard != -1:
            hard_neg_sources.append(u); hard_neg_targets.append(best_w_hard); found_neg_for_this_pos_edge = True
        else: # Fallback
            attempts = 0; max_fallback_attempts = 50
            while attempts < max_fallback_attempts:
                w_fallback = random.randint(0, num_nodes - 1)
                if w_fallback != u and w_fallback != v_pos and tuple(sorted((u, w_fallback))) not in all_true_edges_set:
                    hard_neg_sources.append(u); hard_neg_targets.append(w_fallback); found_neg_for_this_pos_edge = True; break
                attempts += 1
        if not found_neg_for_this_pos_edge: logging.warning(f"HardNegSampler: Could not find any valid negative for positive edge ({u}, {v_pos}).")
    if not hard_neg_sources: return torch.empty(2, 0, dtype=torch.long, device=positive_edges.device)
    return torch.tensor([hard_neg_sources, hard_neg_targets], dtype=torch.long)

def count_common_neighbors(u: int, v: int, adj_matrix: Tensor) -> int: # Add this helper if not present
    if u >= adj_matrix.size(0) or v >= adj_matrix.size(0) or u == v: return 0
    neighbors_u = adj_matrix[u]; neighbors_v = adj_matrix[v]
    return (neighbors_u & neighbors_v).sum().item()


# --- Global Cache & Data Preparation --- (Keep as is)
_CACHED_GRAPH_DATA_AND_SPLITS = None
def prepare_graph_and_splits(mapping_file_path, graph_gexf_path): # ... (Your prepare_graph_and_splits function using generate_hard_negatives_by_common_neighbors) ...
    global _CACHED_GRAPH_DATA_AND_SPLITS
    if _CACHED_GRAPH_DATA_AND_SPLITS is not None:
        logging.info("Using cached graph structure and splits.")
        return _CACHED_GRAPH_DATA_AND_SPLITS
    logging.info("Loading graph structure and preparing splits (once)...")
    num_nodes = 0; edge_index_initial = None
    try:
        with open(mapping_file_path, 'r') as f: mappings = json.load(f)
        num_nodes = len(mappings['id_to_url'])
        nx_graph = nx.read_gexf(graph_gexf_path)
        node_mapping = {node_str: int(node_str) for node_str in nx_graph.nodes()}
        nx_graph = nx.relabel_nodes(nx_graph, node_mapping, copy=True)
        for i in range(num_nodes):
            if i not in nx_graph: nx_graph.add_node(i)
        node_list_for_adj = list(range(num_nodes))
        adj = nx.to_scipy_sparse_array(nx_graph, nodelist=node_list_for_adj, format='coo')
        edge_index_initial = torch.stack([torch.from_numpy(adj.row.astype(np.int64)), torch.from_numpy(adj.col.astype(np.int64))], dim=0)
    except Exception as e: logging.error(f"Failed to load initial graph structure: {e}", exc_info=True); raise

    full_undirected_edge_index = to_undirected(edge_index_initial, num_nodes=num_nodes)
    base_data_topology = Data(edge_index=full_undirected_edge_index, num_nodes=num_nodes)
    all_positive_undirected_edges = base_data_topology.edge_index.clone()

    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False, split_labels=True)
    train_topo, val_topo, test_topo = transform(base_data_topology)
    
    true_edges_set = set()
    if all_positive_undirected_edges.numel() > 0:
        for i in range(all_positive_undirected_edges.size(1)):
            u_true, v_true = all_positive_undirected_edges[0, i].item(), all_positive_undirected_edges[1, i].item()
            true_edges_set.add(tuple(sorted((u_true, v_true))))
    
    adj_matrix_for_neg_sampling = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    if all_positive_undirected_edges.numel() > 0:
        adj_matrix_for_neg_sampling[all_positive_undirected_edges[0], all_positive_undirected_edges[1]] = True
        adj_matrix_for_neg_sampling[all_positive_undirected_edges[1], all_positive_undirected_edges[0]] = True

    for split_data, split_name in [(val_topo, "Validation"), (test_topo, "Test")]:
        if hasattr(split_data, 'pos_edge_label_index') and split_data.pos_edge_label_index is not None and split_data.pos_edge_label_index.numel() > 0:
            logging.info(f"Generating hard negative samples for {split_name} set...")
            hard_neg_edges = generate_hard_negatives_by_common_neighbors(
                split_data.pos_edge_label_index, num_nodes, true_edges_set, adj_matrix_for_neg_sampling, num_candidates_to_rank=50)
            split_data.neg_edge_label_index = hard_neg_edges
            if hard_neg_edges.nelement() > 0: split_data.neg_edge_label = torch.zeros(hard_neg_edges.size(1), dtype=torch.long)
            else: split_data.neg_edge_label = torch.empty(0, dtype=torch.long)
        else:
            logging.warning(f"{split_name} data has no 'pos_edge_label_index' or empty. No hard negatives generated.")
            split_data.neg_edge_label_index = torch.empty(2,0, dtype=torch.long); split_data.neg_edge_label = torch.empty(0, dtype=torch.long)

    if not torch.equal(train_topo.edge_index, val_topo.edge_index): val_topo.edge_index = train_topo.edge_index.clone()
    if not torch.equal(train_topo.edge_index, test_topo.edge_index): test_topo.edge_index = train_topo.edge_index.clone()
    _CACHED_GRAPH_DATA_AND_SPLITS = (train_topo, val_topo, test_topo, all_positive_undirected_edges, num_nodes)
    logging.info("Graph structure, splits, and hard negatives prepared and cached.")
    return _CACHED_GRAPH_DATA_AND_SPLITS

def load_node_features_for_run(embedding_file_path, num_nodes_total): # ... (Keep as is) ...
    logging.info(f"Loading node features from: {embedding_file_path}")
    try:
        embeddings_data = np.load(embedding_file_path)
        node_ids_in_file = embeddings_data['node_ids']; features_in_file = embeddings_data['embeddings']
        logging.info(f"  Raw embeddings shape from file: {features_in_file.shape}")
        node_id_to_file_idx = {node_id: i for i, node_id in enumerate(node_ids_in_file)}
        aligned_features = np.zeros((num_nodes_total, features_in_file.shape[1]), dtype=np.float32)
        nodes_found_count = 0
        for graph_node_id in range(num_nodes_total):
            if graph_node_id in node_id_to_file_idx:
                aligned_features[graph_node_id] = features_in_file[node_id_to_file_idx[graph_node_id]]; nodes_found_count += 1
        logging.info(f"  Aligned features for {nodes_found_count}/{num_nodes_total} nodes. Final shape: {aligned_features.shape}")
        if nodes_found_count == 0 and num_nodes_total > 0 : logging.warning(f"No nodes from graph found in {embedding_file_path}.")
        return torch.tensor(aligned_features, dtype=torch.float)
    except FileNotFoundError: logging.error(f"  Embedding file not found: {embedding_file_path}"); return None
    except Exception as e: logging.error(f"  Error loading embeddings from {embedding_file_path}: {e}", exc_info=True); return None

# --- Training and Evaluation Functions --- (Keep train and test as is)
def train(model, predictor, train_data_obj, optimizer, batch_size, all_positive_edges_for_neg_sampling): # ...
    model.train(); predictor.train()
    node_embeddings_initial = train_data_obj.x.to(DEVICE)
    message_passing_edge_index = train_data_obj.edge_index.to(DEVICE)
    if not hasattr(train_data_obj, 'pos_edge_label_index') or train_data_obj.pos_edge_label_index is None: return 0.0
    pos_train_edge = train_data_obj.pos_edge_label_index.to(DEVICE)
    if pos_train_edge.size(1) == 0: return 0.0
    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(1)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(node_embeddings_initial, message_passing_edge_index)
        edge = pos_train_edge[:, perm]
        pos_out = predictor(h[edge[0]], h[edge[1]])
        neg_edge = negative_sampling(edge_index=all_positive_edges_for_neg_sampling.to(DEVICE),
                                     num_nodes=train_data_obj.num_nodes, num_neg_samples=perm.size(0),
                                     method='sparse').to(DEVICE)
        neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]])
        loss = -torch.log(pos_out + 1e-15).mean() - torch.log(1 - neg_out + 1e-15).mean()
        loss.backward(); optimizer.step()
        total_loss += loss.item() * perm.size(0); total_examples += perm.size(0)
    return total_loss / total_examples if total_examples > 0 else 0.0

@torch.no_grad()
def test(model, predictor, eval_data_obj, batch_size, current_epoch=0): # ...
    model.eval(); predictor.eval()
    node_embeddings_initial = eval_data_obj.x.to(DEVICE)
    message_passing_edge_index = eval_data_obj.edge_index.to(DEVICE)
    h = model(node_embeddings_initial, message_passing_edge_index)
    pos_eval_edge = getattr(eval_data_obj, 'pos_edge_label_index', torch.empty(2,0,dtype=torch.long)).to(DEVICE)
    neg_eval_edge = getattr(eval_data_obj, 'neg_edge_label_index', torch.empty(2,0,dtype=torch.long)).to(DEVICE)
    pos_preds_list = []
    if pos_eval_edge.size(1) > 0:
        for perm in DataLoader(range(pos_eval_edge.size(1)), batch_size):
            edge = pos_eval_edge[:, perm]; pos_preds_list.append(predictor(h[edge[0]], h[edge[1]]).squeeze().cpu())
    pos_pred = torch.cat(pos_preds_list, dim=0) if pos_preds_list else torch.empty(0, device='cpu')
    neg_preds_list = []
    if neg_eval_edge.size(1) > 0:
        for perm in DataLoader(range(neg_eval_edge.size(1)), batch_size):
            edge = neg_eval_edge[:, perm]; neg_preds_list.append(predictor(h[edge[0]], h[edge[1]]).squeeze().cpu())
    neg_pred = torch.cat(neg_preds_list, dim=0) if neg_preds_list else torch.empty(0, device='cpu')
    
    results = {}; y_true_parts, y_score_parts = [], []
    if pos_pred.numel() > 0: y_true_parts.append(torch.ones(pos_pred.size(0))); y_score_parts.append(pos_pred)
    if neg_pred.numel() > 0: y_true_parts.append(torch.zeros(neg_pred.size(0))); y_score_parts.append(neg_pred)
    if not y_true_parts:
        results = {f'Hits@{K_val}': 0.0 for K_val in K_VALS_HITS}; results['AUC'] = 0.0; results['Precision'] = 0.0
        return results
    y_true = torch.cat(y_true_parts); y_score = torch.cat(y_score_parts)
    if y_true.numel() > 1 and len(torch.unique(y_true)) > 1:
        try: results['AUC'] = roc_auc_score(y_true.numpy(), y_score.numpy())
        except ValueError: results['AUC'] = 0.0
    else: results['AUC'] = 0.0
    y_pred_binary = (y_score >= THRESHOLD_FOR_PRECISION).long()
    results['Precision'] = precision_score(y_true.numpy(), y_pred_binary.numpy(), zero_division=0)
    for K_val in K_VALS_HITS:
        if y_score.numel() == 0: hits_at_k = 0.0
        else:
            num_pos_eval = (y_true == 1).sum().item()
            if num_pos_eval == 0: hits_at_k = 1.0 if K_val > 0 and y_score.numel() >= K_val else 0.0 
            else:
                k_for_topk = min(K_val, len(y_score))
                if k_for_topk == 0 : hits_at_k = 0.0
                else:
                    top_k_indices = torch.topk(y_score, k_for_topk).indices
                    hits_at_k = (y_true[top_k_indices] == 1).sum().item() / min(num_pos_eval, K_val) if K_val > 0 and min(num_pos_eval, K_val) > 0 else 0.0
        results[f'Hits@{K_val}'] = hits_at_k
    return results

# --- Plotting Functions ---
def _draw_single_training_curve_on_ax(ax, emb_name, train_epochs, train_losses, 
                                     val_eval_epochs, val_aucs, val_precisions):
    """Helper to draw training curves on a given Matplotlib Axes object."""
    color_loss = 'tab:red'; ax1 = ax
    ax1.set_xlabel('Epoch', fontsize=8)
    ax1.set_ylabel('Training Loss', color=color_loss, fontsize=8)
    ax1.plot(train_epochs, train_losses, color=color_loss, linestyle='-', label='Loss', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=7)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    ax2 = ax1.twinx() 
    color_auc = 'tab:blue'; color_precision = 'tab:green'
    ax2.set_ylabel('Val AUC / Prec', color='black', fontsize=8) # General label for shared axis
    if val_eval_epochs and val_aucs:
        ax2.plot(val_eval_epochs, val_aucs, color=color_auc, linestyle='--', marker='o', markersize=3, label='Val AUC')
    if val_eval_epochs and val_precisions:
        ax2.plot(val_eval_epochs, val_precisions, color=color_precision, linestyle=':', marker='x', markersize=3, label='Val Prec')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=7)
    min_y_metric = 0
    if val_aucs: min_y_metric = min(min_y_metric, min(val_aucs) - 0.05 if val_aucs else 0)
    if val_precisions: min_y_metric = min(min_y_metric, min(val_precisions) - 0.05 if val_precisions else 0)
    ax2.set_ylim(max(0, min_y_metric), 1.05)

    ax1.set_title(emb_name, fontsize=9, pad=10) # Add padding for title
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right', fontsize=6)


def plot_grid_of_training_curves(all_runs_data_for_grid, num_rows, num_cols):
    emb_names_list = list(all_runs_data_for_grid.keys())
    num_plots = len(emb_names_list)

    if num_plots == 0: logging.info("No training curve data to plot in grid."); return

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4.5, num_rows * 3.8), squeeze=False) 
    fig.suptitle("Training Progression for All Embedding Types", fontsize=18, y=1.0) # Adjusted y for suptitle

    for i, emb_name_item in enumerate(emb_names_list):
        row_idx = i // num_cols
        col_idx = i % num_cols
        if row_idx >= num_rows: logging.warning(f"Skipping plot for {emb_name_item}, grid size exceeded."); continue
            
        ax_current = axs[row_idx, col_idx]
        run_data_item = all_runs_data_for_grid[emb_name_item]
        
        _draw_single_training_curve_on_ax(
            ax_current, emb_name_item,
            run_data_item['train_epochs'], run_data_item['train_losses'],
            run_data_item['val_eval_epochs'], run_data_item['val_aucs'], run_data_item['val_precisions']
        )

    for i in range(num_plots, num_rows * num_cols): # Hide unused subplots
        row_idx = i // num_cols; col_idx = i % num_cols
        if row_idx < num_rows and col_idx < num_cols : fig.delaxes(axs[row_idx, col_idx])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # rect to make space for suptitle & bottom labels
    
    plots_dir = "training_plots_per_embedding" # Save grid plot here too or separate
    os.makedirs(plots_dir, exist_ok=True)
    grid_plot_filename = os.path.join(plots_dir, "ALL_training_curves_grid.png")
    plt.savefig(grid_plot_filename, dpi=150)
    logging.info(f"Saved grid of training curves to {grid_plot_filename}")
    plt.close(fig)

def plot_comparison_charts(results_collection, metric_names=['AUC', 'Precision']):
    # ... (plot_comparison_charts as before, no changes needed here if it worked)
    emb_type_names = list(results_collection.keys())
    num_metrics_to_plot = len(metric_names)
    if not emb_type_names or num_metrics_to_plot == 0: logging.warning("No results or metrics for comparison plot."); return

    fig, axs = plt.subplots(num_metrics_to_plot, 1, figsize=(12, 6 * num_metrics_to_plot), squeeze=False)
    try: cmap = plt.get_cmap('viridis')
    except AttributeError: import matplotlib.cm as cm; cmap = cm.get_cmap('viridis')
    
    for i, metric_name in enumerate(metric_names):
        metric_values = [results_collection[name].get(metric_name, 0.0) for name in emb_type_names]
        ax = axs[i, 0]
        colors_for_bars = list(cmap(np.linspace(0.2, 0.9, len(emb_type_names))))
        colors_for_bars[0] = '#FF5733'  # Custom color for first bar
        colors_for_bars[1] = '#33C1FF'  # Custom color for second bar
        bars = ax.bar(emb_type_names, metric_values, color=colors_for_bars)
        ax.set_xlabel("Embedding Type", fontsize=12); ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"Test {metric_name} Comparison", fontsize=14)
        ax.set_xticks(range(len(emb_type_names)))
        ax.set_xticklabels(emb_type_names, rotation=45, ha="right", fontsize=10)
        if metric_name in ['AUC', 'Precision'] or 'Hits@' in metric_name : ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar_idx, bar_item in enumerate(bars): # Renamed bar to bar_item
            yval = bar_item.get_height()
            ax.text(bar_idx, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='medium')
    plt.tight_layout(pad=3.0)
    plt.savefig("embedding_comparison_metrics.png", dpi=150)
    logging.info("Saved embedding comparison plot to embedding_comparison_metrics.png")


# --- Main Training Pipeline ---
if __name__ == "__main__":
    try:
        train_topo_g, val_topo_g, test_topo_g, all_pos_edges_g, num_nodes_g = \
            prepare_graph_and_splits(MAPPING_FILE, GRAPH_FILE)
    except Exception as e:
        logging.error(f"Fatal error during initial data preparation: {e}", exc_info=True); exit()
    if train_topo_g is None: logging.error("Initial data preparation returned None. Exiting."); exit()

    overall_test_results = {}
    all_runs_training_curves_data = {} # For the grid plot

    for emb_name, emb_file_path_current in EMBEDDING_FILES.items():
        logging.info(f"===== PROCESSING EMBEDDING TYPE: {emb_name} =====")
        node_embedding_layer_for_this_run = None; current_node_features = None; input_feature_dim_current = 0

        if emb_name == "LearnableEmb":
            # ... (LearnableEmb setup as before)
            logging.info(f"Using learnable embeddings with dimension: {LEARNABLE_EMB_DIM}")
            node_embedding_layer_for_this_run = torch.nn.Embedding(num_nodes_g, LEARNABLE_EMB_DIM).to(DEVICE)
            torch.nn.init.xavier_uniform_(node_embedding_layer_for_this_run.weight)
            current_node_features = node_embedding_layer_for_this_run.weight 
            input_feature_dim_current = LEARNABLE_EMB_DIM
        else:
            # ... (Pre-trained embedding loading as before)
            if emb_file_path_current is None: logging.error(f"Emb file path is None for {emb_name}. Skipping."); continue
            current_node_features_cpu = load_node_features_for_run(emb_file_path_current, num_nodes_g)
            if current_node_features_cpu is None: logging.warning(f"Skipping {emb_name} due to error."); continue
            current_node_features = current_node_features_cpu
            input_feature_dim_current = current_node_features.size(1)

        current_train_data = train_topo_g.clone(); current_train_data.x = current_node_features
        current_val_data = val_topo_g.clone(); current_val_data.x = current_node_features
        current_test_data = test_topo_g.clone(); current_test_data.x = current_node_features
        logging.info(f"Input feature dimension for {emb_name}: {input_feature_dim_current}")

        gnn = GNNStack(input_dim=input_feature_dim_current, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                       num_layers=NUM_GNN_LAYERS, dropout=DROPOUT, emb=True).to(DEVICE)
        link_predictor = LinkPredictor(in_channels=OUTPUT_DIM, hidden_channels=OUTPUT_DIM, out_channels=1,
                                       num_layers=LINK_PRED_LAYERS, dropout=DROPOUT).to(DEVICE)
        link_predictor.reset_parameters()
        params_to_optimize = list(gnn.parameters()) + list(link_predictor.parameters())
        if node_embedding_layer_for_this_run is not None:
            params_to_optimize.extend(list(node_embedding_layer_for_this_run.parameters()))
        optimizer = torch.optim.Adam(params_to_optimize, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        run_train_losses, run_val_aucs, run_val_precisions, run_val_eval_epochs = [], [], [], []
        logging.info(f"Starting training for {emb_name}...")
        for epoch in range(1, EPOCHS + 1):
            loss = train(gnn, link_predictor, current_train_data, optimizer, BATCH_SIZE, all_pos_edges_g)
            run_train_losses.append(loss)
            if epoch % VAL_EVAL_FREQ == 0 or epoch == EPOCHS:
                gnn.eval(); link_predictor.eval()
                with torch.no_grad():
                    temp_val_results = test(gnn, link_predictor, current_val_data, BATCH_SIZE, current_epoch=epoch)
                run_val_aucs.append(temp_val_results['AUC'])
                run_val_precisions.append(temp_val_results.get('Precision', 0.0))
                run_val_eval_epochs.append(epoch)
                logging.info(f"Emb: {emb_name}, Ep: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {temp_val_results['AUC']:.4f}, Val Precision: {temp_val_results.get('Precision', 0.0):.4f}")
            elif epoch==1 and VAL_EVAL_FREQ !=1 : logging.info(f"Emb: {emb_name}, Ep: {epoch:03d}, Loss: {loss:.4f}")
        
        all_runs_training_curves_data[emb_name] = {
            'train_epochs': list(range(1, EPOCHS + 1)), 'train_losses': run_train_losses,
            'val_eval_epochs': run_val_eval_epochs, 'val_aucs': run_val_aucs, 'val_precisions': run_val_precisions
        }
        
        logging.info(f"Evaluating {emb_name} on Test Set...")
        gnn.eval(); link_predictor.eval()
        with torch.no_grad():
            final_test_results = test(gnn, link_predictor, current_test_data, BATCH_SIZE, current_epoch=EPOCHS)
        overall_test_results[emb_name] = final_test_results
        logging.info(f"Test Results for {emb_name}: AUC={final_test_results['AUC']:.4f}, Precision={final_test_results['Precision']:.4f}, Hits@{K_VALS_HITS[0]}={final_test_results.get(f'Hits@{K_VALS_HITS[0]}', 'N/A')}")

    logging.info("\n===== Overall Comparison of Embeddings (Final Test Metrics) =====")
    for emb_name_key, metrics in overall_test_results.items():
        hits_val_str = metrics.get(f'Hits@{K_VALS_HITS[0]}', 'N/A')
        hits_k0_str = f"{hits_val_str:.4f}" if isinstance(hits_val_str, float) else hits_val_str
        logging.info(f"{emb_name_key}: AUC={metrics.get('AUC',0.0):.4f}, Precision={metrics.get('Precision',0.0):.4f}, Hits@{K_VALS_HITS[0]}={hits_k0_str}")
    
    plot_metric_names_summary = ['AUC', 'Precision']
    if K_VALS_HITS: plot_metric_names_summary.append(f'Hits@{K_VALS_HITS[0]}')
    plot_comparison_charts(overall_test_results, metric_names=plot_metric_names_summary)

    # NEW: Plot grid of all training curves
    num_embedding_types_plot = len(all_runs_training_curves_data)
    if num_embedding_types_plot > 0:
        grid_cols_config = 5  # For a 2x5 grid
        grid_rows_config = (num_embedding_types_plot + grid_cols_config - 1) // grid_cols_config # Calculate rows needed
        plot_grid_of_training_curves(all_runs_training_curves_data, grid_rows_config, grid_cols_config)
    else:
        logging.info("No data available to plot grid of training curves.")

    logging.info("===== Pipeline Complete =====")