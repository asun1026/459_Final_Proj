from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    test_size: float = 0.2
    neg_multiplier_train: int = 1
    neg_multiplier_test: int = 100
    pca_components: int | None = 128
    random_state: int = 42

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["node_ids"], data["embeddings"]


def node_disjoint_edge_split(edges, test_frac, rng):
    rng.shuffle(edges)
    train, test, test_nodes = [], [], set()
    target = math.ceil(len(edges) * test_frac)
    for u, v in edges:
        if len(test) >= target or u in test_nodes or v in test_nodes:
            train.append((u, v))
        else:
            test.append((u, v))
            test_nodes.update([u, v])
    return train, test


def sample_random_negatives(graph: nx.Graph, n: int, rng: random.Random):
    nodes = list(graph.nodes())
    neg = set()
    while len(neg) < n:
        u, v = rng.sample(nodes, 2)
        if not graph.has_edge(u, v) and (u, v) not in neg and (v, u) not in neg:
            neg.add((u, v))
    return list(neg)


def make_features(emb, node_ids, edges):
    idx = {nid: i for i, nid in enumerate(node_ids)}
    return np.asarray([
        np.concatenate([emb[idx[u]], emb[idx[v]]])
        for u, v in edges if u in idx and v in idx
    ])


def supervised_lp(emb, node_ids, tr_pos, tr_neg, te_pos, te_neg):
    X_tr = np.vstack([make_features(emb, node_ids, tr_pos), make_features(emb, node_ids, tr_neg)])
    y_tr = np.concatenate([np.ones(len(tr_pos)), np.zeros(len(tr_neg))])
    X_te = np.vstack([make_features(emb, node_ids, te_pos), make_features(emb, node_ids, te_neg)])
    y_te = np.concatenate([np.ones(len(te_pos)), np.zeros(len(te_neg))])
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=1)
    clf.fit(X_tr, y_tr)
    scores = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, scores), average_precision_score(y_te, scores)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation routine
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_embeddings(graph: nx.Graph, emb_path: str, cfg: EvalConfig):
    rng = random.Random(cfg.random_state)
    edges = list(graph.edges())
    train_edges, test_edges = node_disjoint_edge_split(edges, cfg.test_size, rng)
    G_train = graph.copy()
    G_train.remove_edges_from(test_edges)

    node_ids, emb = load_embeddings(emb_path)
    if cfg.pca_components and emb.shape[1] > cfg.pca_components:
        mask = np.isin(node_ids, list(G_train.nodes()))
        pca = PCA(n_components=cfg.pca_components, random_state=cfg.random_state)
        pca.fit(emb[mask])
        emb = pca.transform(emb)
        logging.info("PCA fitted on %d training nodes", int(mask.sum()))

    tr_neg = sample_random_negatives(G_train, len(train_edges) * cfg.neg_multiplier_train, rng)
    te_neg = sample_random_negatives(G_train, len(test_edges) * cfg.neg_multiplier_test, rng)
    return supervised_lp(emb, node_ids, train_edges, tr_neg, test_edges, te_neg)

# ─────────────────────────────────────────────────────────────────────────────
# Main – evaluate multiple layers & generate tables / charts
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    from DataExtract.graph_builder import build_graph_from_jsonl

    jsonl_path = "DataExtract/Data/stackexchange_cdxtoolkit_data_all_fixed.jsonl"
    embedding_paths = {
        "SBERT": "LLM_embed/embed_data/node_sbert_embeddings.npz",
        "Layer_04": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_04.npz",
        "Layer_08": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_08.npz",
        "Layer_12": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_12.npz",
        "Layer_16": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_16.npz",
        "Layer_20": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_20.npz",
        "Layer_24": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_24.npz",
        "Layer_28": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_28.npz",
        "Layer_32": "LLM_embed/embed_data/node_hidden_state_embeddings_DeepSeek-R1-Distill-Llama-8B_layer_32.npz",
    }

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(jsonl_path)
    G, *_ = build_graph_from_jsonl(jsonl_path)
    logging.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    cfg = EvalConfig()
    records: List[dict] = []

    for name, path in embedding_paths.items():
        if not os.path.exists(path):
            logging.warning("Missing embedding file: %s", path)
            continue
        logging.info("Evaluating %s", name)
        auc, ap = evaluate_embeddings(G, path, cfg)
        records.append({"Model": name, "AUC‑ROC": auc, "Average Precision": ap})
        logging.info("%s – AUC: %.4f,  AP: %.4f", name, auc, ap)

    if not records:
        raise SystemExit("No evaluations completed.")

    df = pd.DataFrame(records).sort_values("Model").reset_index(drop=True)
    print("\n=== Link‑Prediction Results ===")
    print(df.to_string(index=False, float_format="{:.4f}".format))

    df.to_latex("results_table.tex", index=False, float_format="%.4f", caption="Link‑prediction performance (higher is better)", label="tab:link_pred")
    logging.info("LaTeX table written to results_table.tex")

    # ---------------------------- Separate charts -----------------------------
    def plot_metric(column: str, outfile: str, title: str):
        plt.figure(figsize=(10, 4))
        plt.bar(df["Model"], df[column])
        plt.ylabel(column)
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        logging.info("Chart saved to %s", outfile)
        plt.close()

    plot_metric("AUC‑ROC", "auc_roc.png", "AUC‑ROC by model/layer")
    plot_metric("Average Precision", "average_precision.png", "Average Precision by model/layer")
