#!/usr/bin/env python
# -*- coding: utf-8 -*-
# graphSAGE3.py

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
import torch_geometric as pyg
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import negative_sampling, train_test_split_edges, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
# Data Paths
DATA_EXTRACT_DIR = "DataExtract/Data"
LLM_EMBED_DIR = "LLM_embed/embed_data"
GRAPH_FILE = os.path.join(DATA_EXTRACT_DIR, "stackexchange_graph.gexf") # GEXF from your graph_builder
MAPPING_FILE = os.path.join(DATA_EXTRACT_DIR, "url_id_mapping.json")
NODE_TEXT_DATA_FILE = os.path.join(DATA_EXTRACT_DIR, "node_text_data.json") # Used for getting node count if graph not loaded directly

# Choose ONE embedding file
# Example: replace with the actual layer and model name you want to use
EMBEDDING_MODEL_SHORT_NAME = 'DeepSeek-R1-Distill-Llama-8B' # From your llm_embed.py
EMBEDDING_LAYER_IDX = 28 # Example layer, choose from your TARGET_LAYER_INDICES
EMBEDDING_FILE = os.path.join(LLM_EMBED_DIR, f"node_hidden_state_embeddings_{EMBEDDING_MODEL_SHORT_NAME}_layer_{str(EMBEDDING_LAYER_IDX).zfill(2)}.npz")

# GNN Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 5e-7
WEIGHT_DECAY = 0 # 5e-4
EPOCHS = 200
BATCH_SIZE = 2048 # Adjust based on your graph size and memory
HIDDEN_DIM = 128  # GNN hidden dimension
OUTPUT_DIM = 64   # GNN output dimension (embedding dim before link predictor)
NUM_GNN_LAYERS = 2
DROPOUT = 0.3
LINK_PRED_LAYERS = 3 # Number of layers in the LinkPredictor MLP

# Evaluation
K_VALS_HITS = [10, 50, 100] # K values for Hits@K

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gnn_link_pred_stackexchange.log"),
        logging.StreamHandler()
    ]
)
logging.info(f"Using device: {DEVICE}")

# --- GNN Model Definitions (Adapted from your example) ---
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
        super(GNNStack, self).__init__()
        conv_model = pyg.nn.SAGEConv # Using SAGEConv as in example

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(conv_model(input_dim, output_dim))
        else:
            self.convs.append(conv_model(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(conv_model(hidden_dim, hidden_dim))
            self.convs.append(conv_model(hidden_dim, output_dim))

        self.dropout_rate = dropout # Renamed from self.dropout to avoid conflict with nn.Dropout module
        self.num_layers = num_layers
        self.emb = emb # If true, forward returns embeddings, else log_softmax for classification

        # Post-message-passing processing (Not strictly needed if emb=True and output_dim is final embedding dim)
        # self.post_mp = nn.Sequential(
        #     nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Dropout(self.dropout_rate),
        #     nn.Linear(output_dim, output_dim)
        # )

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1: # No ReLU + Dropout on last layer if outputting embeddings
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # if self.emb:
        #     return self.post_mp(x) # Apply post_mp if emb=True
        # return F.log_softmax(self.post_mp(x), dim=1)
        if self.emb:
             return x # Return embeddings directly
        return F.log_softmax(x, dim=1) # Or log_softmax if not returning embeddings

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout_rate = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j # Element-wise product
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1: # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return torch.sigmoid(x) # Sigmoid for probability output

# --- Data Loading and Preprocessing ---
def load_data():
    logging.info("Loading graph data...")
    try:
        # Load mappings
        with open(MAPPING_FILE, 'r') as f:
            mappings = json.load(f)
        id_to_url = {int(k): v for k,v in mappings['id_to_url'].items()} # Ensure keys are int
        num_nodes = len(id_to_url)
        logging.info(f"Loaded mappings for {num_nodes} nodes.")

        # Load graph from GEXF
        # Important: Ensure node IDs in GEXF are integers and match your 'id_to_url' keys
        nx_graph = nx.read_gexf(GRAPH_FILE)
        # Convert node labels to integers if they are strings in GEXF
        try:
            nx_graph = nx.relabel_nodes(nx_graph, {node_label: int(node_label) for node_label in nx_graph.nodes()})
        except ValueError:
            logging.warning("Could not convert all GEXF node labels to int. Assuming they are already correct.")
            pass

        # Ensure all nodes from mapping are in the graph, add if missing
        for i in range(num_nodes):
            if i not in nx_graph:
                nx_graph.add_node(i)

        logging.info(f"Loaded NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges.")
        if nx_graph.number_of_nodes() != num_nodes:
            logging.warning(f"Mismatch: Mapped nodes {num_nodes}, Graph nodes {nx_graph.number_of_nodes()}. Using {num_nodes} as reference.")

        # Create edge_index for PyG
        # Your graph is directed. For GraphSAGE, it's common to make it undirected
        # for message passing, or handle directedness explicitly if the model supports it.
        # Here, we'll make it undirected for simplicity with SAGEConv.
        adj = nx.to_scipy_sparse_array(nx_graph, nodelist=sorted(nx_graph.nodes()), format='coo')
        row = torch.from_numpy(adj.row.astype(np.int64))
        col = torch.from_numpy(adj.col.astype(np.int64))
        edge_index = torch.stack([row, col], dim=0)
        # Original edges (for supervision if needed, though train_test_split_edges will handle this)
        # For link prediction, PyG's train_test_split_edges expects undirected edges.
        # We can convert to undirected first, then split.
        # However, if your original links are what you want to predict, it's better to use them.
        # For simplicity, let's assume we predict on the given directed edges.
        # We might need to adjust how train_test_split_edges works or do it manually.

    except FileNotFoundError as e:
        logging.error(f"Error loading graph data: {e}. Please check paths.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading graph: {e}", exc_info=True)
        return None


    logging.info(f"Loading embeddings from {EMBEDDING_FILE}...")
    try:
        embeddings_data = np.load(EMBEDDING_FILE)
        node_ids_emb = embeddings_data['node_ids']
        features = embeddings_data['embeddings']
        logging.info(f"Loaded embeddings shape: {features.shape}")

        # Create a mapping from node_id to its index in the embeddings_data
        node_id_to_emb_idx = {node_id: i for i, node_id in enumerate(node_ids_emb)}

        # Align features with the graph's node ordering (0 to num_nodes-1)
        # Initialize with zeros or random for nodes not in embeddings
        aligned_features = np.zeros((num_nodes, features.shape[1]), dtype=np.float32)
        nodes_with_embeddings = 0
        for graph_node_id in range(num_nodes):
            if graph_node_id in node_id_to_emb_idx:
                aligned_features[graph_node_id] = features[node_id_to_emb_idx[graph_node_id]]
                nodes_with_embeddings += 1
            else:
                logging.warning(f"Node ID {graph_node_id} from graph not found in embeddings file. Using zero vector.")
        logging.info(f"Aligned features for {nodes_with_embeddings}/{num_nodes} nodes.")
        node_features = torch.tensor(aligned_features, dtype=torch.float)

    except FileNotFoundError:
        logging.error(f"Embedding file not found: {EMBEDDING_FILE}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading embeddings: {e}", exc_info=True)
        return None

    # Create a base Data object. For RandomLinkSplit, it's best to provide the graph
    # structure you intend to split. If predicting on an undirected basis (common for SAGEConv),
    # provide the full undirected graph.
    # If your task is strictly directed link prediction, it's more complex;
    # For now, assuming undirected link prediction task for SAGEConv.
    full_undirected_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    base_data = Data(x=node_features, edge_index=full_undirected_edge_index)

    all_positive_undirected_edges = base_data.edge_index.clone() # Use .clone() for safety

    logging.info("Splitting edges using RandomLinkSplit...")
    # Configure RandomLinkSplit
    # is_undirected=True: The input graph (base_data.edge_index) is undirected.
    # add_negative_train_samples=False: We'll do negative sampling manually in the train loop
    #                                   to maintain control and ensure we avoid all true positive edges.
    # split_labels=True: This is crucial. It provides:
    #                    - edge_label: Tensor of 0s and 1s for supervision edges.
    #                    - edge_label_index: Tensor of shape [2, num_supervision_edges] containing
    #                                        both positive and negative supervision edges.
    # neg_sampling_ratio=1.0 (default for val/test): For each positive val/test edge,
    #                                                one negative edge is sampled.
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False, # We will do manual negative sampling in train loop
        split_labels=True
    )

    try:
        train_split_data, val_split_data, test_split_data = transform(base_data)
    except Exception as e:
        logging.error(f"Error during RandomLinkSplit: {e}", exc_info=True)
        return None, None, None # Return three Nones
    
    true_edges_set = set()
    for i in range(all_positive_undirected_edges.size(1)):
        u_true, v_true = all_positive_undirected_edges[0, i].item(), all_positive_undirected_edges[1, i].item()
        true_edges_set.add(tuple(sorted((u_true, v_true))))

    # Create dense adjacency matrix from all_positive_undirected_edges for neighbor lookups
    # Ensure device consistency if using CUDA later; for now, CPU is fine for this setup part.
    adj_matrix_for_neg_sampling = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adj_matrix_for_neg_sampling[all_positive_undirected_edges[0], all_positive_undirected_edges[1]] = True
    adj_matrix_for_neg_sampling[all_positive_undirected_edges[1], all_positive_undirected_edges[0]] = True # Ensure symmetric for undirected

    # Number of negative samples per positive for val/test (can be a parameter)
    # RandomLinkSplit with split_labels=True uses neg_sampling_ratio=1.0 by default for val/test,
    # so if we want to match that count:
    num_neg_val = val_split_data.pos_edge_label_index.size(1) # Match number of positives
    num_neg_test = test_split_data.pos_edge_label_index.size(1)

    
    # We will replace the neg_edge_label_index from RandomLinkSplit
    val_hard_neg_edges = generate_hard_negative_set(
        val_split_data.pos_edge_label_index,
        num_nodes,
        true_edges_set,
        adj_matrix_for_neg_sampling,
        num_neg_per_pos=1 # Can be increased, but ensure the generation logic handles it well for distinctness
    )
    val_split_data.neg_edge_label_index = val_hard_neg_edges
    if val_hard_neg_edges.nelement() > 0 : # Check if tensor is not empty
         val_split_data.neg_edge_label = torch.zeros(val_hard_neg_edges.size(1), dtype=torch.long)
    else: # Handle empty tensor case
         val_split_data.neg_edge_label = torch.empty(0, dtype=torch.long)


    logging.info("Generating hard negative samples for test set...")
    test_hard_neg_edges = generate_hard_negative_set(
        test_split_data.pos_edge_label_index,
        num_nodes,
        true_edges_set,
        adj_matrix_for_neg_sampling,
        num_neg_per_pos=1
    )
    test_split_data.neg_edge_label_index = test_hard_neg_edges
    if test_hard_neg_edges.nelement() > 0:
        test_split_data.neg_edge_label = torch.zeros(test_hard_neg_edges.size(1), dtype=torch.long)
    else:
        test_split_data.neg_edge_label = torch.empty(0, dtype=torch.long)

    # Log information about the splits (Corrected to use *_index for edge counts)
    for split_name, split_data in [("Train", train_split_data), ("Validation", val_split_data), ("Test", test_split_data)]:
        logging.info(f"{split_name} data: Nodes={split_data.num_nodes}, FeaturesShape={split_data.x.shape}")
        if hasattr(split_data, 'edge_index') and split_data.edge_index is not None:
             logging.info(f"  {split_name} MP Edges: {split_data.edge_index.size(1)}")
        else:
             logging.warning(f"  {split_name} data has no 'edge_index' or it's None for MP.")

        num_pos_sup_edges = 0
        # Use pos_edge_label_index for the EDGES
        if hasattr(split_data, 'pos_edge_label_index') and split_data.pos_edge_label_index is not None:
            num_pos_sup_edges = split_data.pos_edge_label_index.size(1) # Corrected: use _index
            logging.info(f"  {split_name} Positive Supervision Edges: {num_pos_sup_edges}")
        else:
            logging.warning(f"  {split_name} data has no 'pos_edge_label_index' or it's None.")

        num_neg_sup_edges = 0
        # For train_split_data, neg_edge_label_index might not exist if add_negative_train_samples=False
        # However, RandomLinkSplit with split_labels=True usually creates neg_edge_label_index for val/test
        # with neg_sampling_ratio=1.0 by default.
        if hasattr(split_data, 'neg_edge_label_index') and split_data.neg_edge_label_index is not None:
            num_neg_sup_edges = split_data.neg_edge_label_index.size(1) # Corrected: use _index
            logging.info(f"  {split_name} Negative Supervision Edges: {num_neg_sup_edges}")
        else:
            logging.warning(f"  {split_name} data missing 'neg_edge_label_index' or it's None.")
        logging.info(f"  {split_name} Total (Positive) Supervision Edges from attributes: {num_pos_sup_edges}")
        # Total supervision edges would be num_pos_sup_edges + num_neg_sup_edges if both are present


    # Sanity check and enforcement for transductive message passing edges
    if not torch.equal(train_split_data.edge_index, val_split_data.edge_index):
        logging.warning("Adjusting val_split_data.edge_index to match train_split_data.edge_index.")
        val_split_data.edge_index = train_split_data.edge_index
    if not torch.equal(train_split_data.edge_index, test_split_data.edge_index):
        logging.warning("Adjusting test_split_data.edge_index to match train_split_data.edge_index.")
        test_split_data.edge_index = train_split_data.edge_index

    return train_split_data, val_split_data, test_split_data, all_positive_undirected_edges

def get_two_hop_non_neighbors(node_u: int, num_nodes: int, adj_matrix: Tensor, all_true_edges_set: set) -> list:
    """
    Finds nodes that are 2 hops away from node_u, are not node_u itself,
    are not 1-hop neighbors of node_u, and do not form a true edge with node_u.
    """
    if node_u >= adj_matrix.size(0):
        return []

    one_hop_u = adj_matrix[node_u] > 0
    
    # Calculate 2-hop reachability for node_u efficiently
    # (A @ A_u_col)_i gives sum over k of A_ik * A_ku.
    # If A_u_col is a column vector representing neighbors of u, then (A @ A_u_col) gives nodes reachable from u's neighbors.
    # More simply: find neighbors of u's neighbors.
    
    # Get 1-hop neighbors of u
    u_neighbors_indices = one_hop_u.nonzero(as_tuple=False).view(-1)
    
    potential_two_hop_nodes = set()
    for neighbor_idx in u_neighbors_indices:
        # Get neighbors of this neighbor
        neighbors_of_neighbor = (adj_matrix[neighbor_idx.item()] > 0).nonzero(as_tuple=False).view(-1)
        for two_hop_node_idx in neighbors_of_neighbor:
            potential_two_hop_nodes.add(two_hop_node_idx.item())
            
    hard_negative_candidates = []
    for w_candidate in potential_two_hop_nodes:
        # Must not be u, not a 1-hop neighbor of u, and (u, w_candidate) must not be a true edge
        if w_candidate != node_u and \
           not one_hop_u[w_candidate] and \
           tuple(sorted((node_u, w_candidate))) not in all_true_edges_set:
            hard_negative_candidates.append(w_candidate)
            
    return hard_negative_candidates

def generate_hard_negative_set(positive_edges: Tensor, num_nodes: int,
                               all_true_edges_set: set, adj_matrix: Tensor,
                               num_neg_per_pos: int = 1) -> Tensor:
    """
    Generates a set of hard negative edges corresponding to the given positive edges.
    Aims for one hard negative for each source node in positive_edges.
    """
    hard_neg_sources = []
    hard_neg_targets = []
    
    source_nodes_in_positive_set = torch.unique(positive_edges[0])

    if source_nodes_in_positive_set.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=positive_edges.device)

    for u_node_idx_tensor in source_nodes_in_positive_set:
        u = u_node_idx_tensor.item()
        
        # Try to find num_neg_per_pos hard negatives for this u
        found_neg_for_this_u = 0
        
        # Get 2-hop non-neighbor candidates
        hard_candidates_for_u = get_two_hop_non_neighbors(u, num_nodes, adj_matrix, all_true_edges_set)
        
        if hard_candidates_for_u:
            random.shuffle(hard_candidates_for_u) # Shuffle to pick different ones if needed
            for w_hard in hard_candidates_for_u:
                if found_neg_for_this_u < num_neg_per_pos:
                    hard_neg_sources.append(u)
                    hard_neg_targets.append(w_hard)
                    found_neg_for_this_u += 1
                else:
                    break
        
        # Fallback: If not enough 2-hop hard negatives found, sample random negatives for u
        # (ensuring they are not true edges and not u itself)
        # More robust random sampling is needed for multiple distinct negatives per u.
        # For num_neg_per_pos = 1, this fallback is simpler.
        attempts = 0
        max_random_attempts = num_nodes * 2 # Max attempts to find a random negative for this u
        
        while found_neg_for_this_u < num_neg_per_pos and attempts < max_random_attempts:
            w_random = random.randint(0, num_nodes - 1)
            if w_random != u and tuple(sorted((u, w_random))) not in all_true_edges_set:
                # Avoid adding duplicates if we already added this random w for this u
                is_new_neg = True
                if num_neg_per_pos > 1: # Check if (u, w_random) was already added
                    for i in range(len(hard_neg_sources)):
                        if hard_neg_sources[i] == u and hard_neg_targets[i] == w_random:
                            is_new_neg = False
                            break
                if is_new_neg:
                    hard_neg_sources.append(u)
                    hard_neg_targets.append(w_random)
                    found_neg_for_this_u += 1
            attempts += 1
            
        if found_neg_for_this_u == 0 and num_neg_per_pos > 0:
            # Last resort if u is highly connected or graph is dense and no neg found easily.
            # This indicates a potential issue or very challenging node u.
            # For now, we might not add any negative for this u if all attempts fail.
            logging.warning(f"Could not find any valid negative sample for source node {u} after all attempts.")


    if not hard_neg_sources: # If no negatives were generated at all
        return torch.empty(2, 0, dtype=torch.long, device=positive_edges.device)
        
    return torch.tensor([hard_neg_sources, hard_neg_targets], dtype=torch.long)

# --- Training and Evaluation Functions ---
def train(model, predictor, train_data_obj, optimizer, batch_size, all_positive_edges_for_neg_sampling):
    model.train()
    predictor.train()

    node_embeddings_initial = train_data_obj.x.to(DEVICE)
    message_passing_edge_index = train_data_obj.edge_index.to(DEVICE)

    # Positive supervision edges for training from pos_edge_label_index
    if not hasattr(train_data_obj, 'pos_edge_label_index') or train_data_obj.pos_edge_label_index is None: # Corrected attribute
        logging.error("Train data is missing 'pos_edge_label_index'. Cannot proceed with training.")
        return 0.0
    pos_train_edge = train_data_obj.pos_edge_label_index.to(DEVICE) # Corrected attribute

    if pos_train_edge.size(1) == 0:
        logging.warning("No positive training supervision edges found. Skipping training for this epoch.")
        return 0.0

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(1)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(node_embeddings_initial, message_passing_edge_index)
        edge = pos_train_edge[:, perm]
        pos_out = predictor(h[edge[0]], h[edge[1]])

        neg_edge = negative_sampling(
            edge_index=all_positive_edges_for_neg_sampling.to(DEVICE),
            num_nodes=train_data_obj.num_nodes,
            num_neg_samples=perm.size(0),
            method='sparse').to(DEVICE)
        neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]])

        loss = -torch.log(pos_out + 1e-15).mean() - torch.log(1 - neg_out + 1e-15).mean()
        loss.backward()
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples if total_examples > 0 else 0.0


@torch.no_grad()
def test(model, predictor, eval_data_obj, batch_size, current_epoch=0):
    model.eval()
    predictor.eval()

    node_embeddings_initial = eval_data_obj.x.to(DEVICE)
    message_passing_edge_index = eval_data_obj.edge_index.to(DEVICE)
    h = model(node_embeddings_initial, message_passing_edge_index)

    # Positive supervision edges from pos_edge_label_index
    if not hasattr(eval_data_obj, 'pos_edge_label_index') or eval_data_obj.pos_edge_label_index is None:
        logging.warning(f"Eval data (epoch {current_epoch}) is missing 'pos_edge_label_index'.")
        pos_eval_edge = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    else:
        pos_eval_edge = eval_data_obj.pos_edge_label_index.to(DEVICE) # Corrected attribute

    # Negative supervision edges from neg_edge_label_index
    if not hasattr(eval_data_obj, 'neg_edge_label_index') or eval_data_obj.neg_edge_label_index is None:
        logging.warning(f"Eval data (epoch {current_epoch}) is missing 'neg_edge_label_index'.")
        neg_eval_edge = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    else:
        neg_eval_edge = eval_data_obj.neg_edge_label_index.to(DEVICE) # Corrected attribute


    pos_preds = []
    if pos_eval_edge.size(1) > 0:
        for perm in DataLoader(range(pos_eval_edge.size(1)), batch_size):
            edge = pos_eval_edge[:, perm]
            pos_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_pred = torch.cat(pos_preds, dim=0) if pos_preds else torch.empty(0, device='cpu') # ensure on cpu

    neg_preds = []
    if neg_eval_edge.size(1) > 0:
        for perm in DataLoader(range(neg_eval_edge.size(1)), batch_size):
            edge = neg_eval_edge[:, perm]
            neg_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_pred = torch.cat(neg_preds, dim=0) if neg_preds else torch.empty(0, device='cpu') # ensure on cpu

    # Debug logging (added current_epoch)
    if current_epoch <= 1 or current_epoch % 20 == 0 : # Log for early epochs and periodically
        logging.info(f"--- Debug eval_data_obj @ Epoch {current_epoch} ---")
        if pos_pred.numel() > 0:
            logging.info(f"Sample positive preds: {pos_pred[:min(5, pos_pred.size(0))]}")
            logging.info(f"Mean positive pred: {pos_pred.mean():.4f}, Std positive pred: {pos_pred.std():.4f}")
        else:
            logging.info("No positive edges to predict in this eval split.")
        if neg_pred.numel() > 0:
            logging.info(f"Sample negative preds: {neg_pred[:min(5, neg_pred.size(0))]}")
            logging.info(f"Mean negative pred: {neg_pred.mean():.4f}, Std negative pred: {neg_pred.std():.4f}")
        else:
            logging.info("No negative edges to predict in this eval split.")


    results = {}
    # Calculate ROC AUC
    if pos_pred.numel() == 0 and neg_pred.numel() == 0:
        # ... (handle empty predictions) ...
        results = {f'Hits@{K_val}': 0.0 for K_val in K_VALS_HITS}
        results['AUC'] = 0.0
        return results
    # ... (construct y_true, y_score carefully based on non-empty pos_pred/neg_pred) ...
    # (The existing logic for constructing y_true and y_score needs to handle cases where one is empty)
    y_true_parts = []
    y_score_parts = []
    if pos_pred.numel() > 0:
        y_true_parts.append(torch.ones(pos_pred.size(0)))
        y_score_parts.append(pos_pred)
    if neg_pred.numel() > 0:
        y_true_parts.append(torch.zeros(neg_pred.size(0)))
        y_score_parts.append(neg_pred)

    if not y_true_parts: # Should have been caught by the earlier check
        results = {f'Hits@{K_val}': 0.0 for K_val in K_VALS_HITS}
        results['AUC'] = 0.0
        return results

    y_true = torch.cat(y_true_parts)
    y_score = torch.cat(y_score_parts)


    if y_true.numel() > 1 and len(torch.unique(y_true)) > 1:
        try:
            results['AUC'] = roc_auc_score(y_true.numpy(), y_score.numpy())
        except ValueError as e:
            logging.warning(f"Could not compute AUC for split at epoch {current_epoch}: {e}.")
            results['AUC'] = 0.0
    else:
        results['AUC'] = 0.0

    # Simplified Hits@K
    for K_val in K_VALS_HITS:
        # ... (Hits@K logic, ensure y_true and y_score are correctly formed) ...
        if y_score.numel() == 0: hits_at_k = 0.0
        else:
            num_pos_eval = (y_true == 1).sum().item()
            if num_pos_eval == 0: hits_at_k = 0.0
            else:
                # Ensure K_val is not larger than the number of available scores
                k_for_topk = min(K_val, len(y_score))
                if k_for_topk == 0 : # Handle case where y_score is empty
                    hits_at_k = 0.0
                else:
                    top_k_indices = torch.topk(y_score, k_for_topk).indices
                    hits_at_k = (y_true[top_k_indices] == 1).sum().item() / min(num_pos_eval, K_val) if K_val > 0 else 0.0 # Avoid div by zero if K_val is 0
        results[f'Hits@{K_val}'] = hits_at_k
    return results


# --- Main Training Pipeline ---
if __name__ == "__main__":
    # load_data now returns three Data objects and all_positive_edges
    train_data, val_data, test_data, all_positive_undirected_edges = load_data()

    if train_data is None: # Check if any of them is None if load_data can fail partially
        logging.error("Failed to load data. Exiting.")
        exit()

    logging.info(f"Data loaded. Train data: {train_data}")
    logging.info(f"Val data: {val_data}")
    logging.info(f"Test data: {test_data}")
    logging.info(f"All positive undirected edges for neg sampling: {all_positive_undirected_edges.shape}")


    input_feature_dim = train_data.x.size(1) # Assuming x is same across splits

    gnn = GNNStack(
        input_dim=input_feature_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_GNN_LAYERS,
        dropout=DROPOUT,
        emb=True
    ).to(DEVICE)

    link_predictor = LinkPredictor(
        in_channels=OUTPUT_DIM,
        hidden_channels=OUTPUT_DIM,
        out_channels=1,
        num_layers=LINK_PRED_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(link_predictor.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_losses = []
    val_aucs = []
    val_eval_epochs_list = [] # <<< NEW: Initialize list to store epochs of validation
    test_aucs = [] # If you plan to plot this over epochs (usually test is final)
    val_hits_at_k = {k_hit: [] for k_hit in K_VALS_HITS} # Ensure K_VALS_HITS is defined


    logging.info("Evaluating on Validation Set (BEFORE Training)...")
    # Ensure models are in eval mode for this initial test
    gnn.eval()
    link_predictor.eval()
    with torch.no_grad(): # Explicitly use no_grad for pre-training evaluation
        initial_val_results = test(gnn, link_predictor, val_data, BATCH_SIZE)
    logging.info(f"Initial Val AUC (Untrained Model): {initial_val_results['AUC']:.4f}")
    for k_hit in K_VALS_HITS:
        logging.info(f"Initial Val Hits@{k_hit} (Untrained Model): {initial_val_results[f'Hits@{k_hit}']:.4f}")

    logging.info("Starting training...")
    start_time_total = time.time()

    for epoch in range(1, EPOCHS + 1): # EPOCHS needs to be defined (e.g., EPOCHS = 200)
        epoch_start_time = time.time()
        # Pass all_positive_undirected_edges to train for robust negative sampling
        loss = train(gnn, link_predictor, train_data, optimizer, BATCH_SIZE, all_positive_undirected_edges)
        train_losses.append(loss)
        epoch_duration = time.time() - epoch_start_time

        # Evaluate on validation set
        if epoch % 5 == 0 or epoch == EPOCHS: # Evaluate every 5 epochs or on the last epoch
            val_results = test(gnn, link_predictor, val_data, BATCH_SIZE, current_epoch=epoch)
            val_auc = val_results['AUC']
            val_aucs.append(val_auc)
            val_eval_epochs_list.append(epoch) # <<< ADD THIS: Record the epoch number
            
            for k_hit_val in K_VALS_HITS: # Use a different variable name for clarity
                val_hits_at_k[k_hit_val].append(val_results[f'Hits@{k_hit_val}'])

            logging.info(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, "
                         f"Val Hits@{K_VALS_HITS[0]}: {val_results[f'Hits@{K_VALS_HITS[0]}']:.4f} (Epoch time: {epoch_duration:.2f}s)")
        else:
            logging.info(f"Epoch: {epoch:03d}, Loss: {loss:.4f} (Epoch time: {epoch_duration:.2f}s)")

    total_training_time = time.time() - start_time_total
    logging.info(f"Training finished. Total time: {total_training_time:.2f} seconds.")

    # Evaluate on test set (usually done once after all training)
    logging.info("Evaluating on Test Set...")
    test_results = test(gnn, link_predictor, test_data, BATCH_SIZE, current_epoch=EPOCHS) # Pass test_data
    logging.info(f"Test AUC: {test_results['AUC']:.4f}")
    for k_hit_val in K_VALS_HITS:
        logging.info(f"Test Hits@{k_hit_val}: {test_results[f'Hits@{k_hit_val}']:.4f}")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    # Plot against actual epoch numbers if desired, otherwise against iteration number
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss') # Assumes train_losses has EPOCHS elements
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # Plot validation AUC and Hits@K
    plt.subplot(1, 2, 2)
    if val_aucs: # Check if there's anything to plot
        plt.plot(val_eval_epochs_list, val_aucs, label='Validation AUC', marker='o') # <<< USE THE DYNAMIC LIST
        for k_hit_val in K_VALS_HITS:
             if val_hits_at_k[k_hit_val]: # Check if list is not empty for this K
                plt.plot(val_eval_epochs_list, val_hits_at_k[k_hit_val], label=f'Validation Hits@{k_hit_val}', marker='x') # <<< USE THE DYNAMIC LIST
    else:
        logging.warning("No validation metrics recorded to plot.")

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend() # Call legend after all plots for this subplot are done

    plt.tight_layout()
    plt.savefig("training_metrics_stackexchange.png")
    logging.info("Saved training metrics plot to training_metrics_stackexchange.png")
    # plt.show() # Comment out or ensure your environment supports GUI display if running remotely