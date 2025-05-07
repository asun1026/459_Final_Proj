# Semantic Link Prediction in Web Graphs using LLM Embeddings and GraphSAGE

This project investigates the enhancement of link prediction in large-scale web graphs by integrating rich semantic information derived from Large Language Model (LLM) embeddings. We hypothesize that the nuanced understanding of textual content captured by advanced LLMs can significantly complement traditional graph-based structural features, leading to more accurate discovery of existing (unobserved) and potential future relationships between web pages.

The core methodology involves:
* Constructing a web graph from the Common Crawl dataset, where nodes are web pages and edges are hyperlinks.
* Extracting textual content for each page and generating various types of embeddings:
    * Hidden state representations from an advanced LLM (DeepSeek-R1-Distill-Llama-8B across multiple layers).
    * Embeddings from Sentence-BERT (SBERT) as a strong baseline.
    * Learnable embeddings trained from scratch as a structural baseline.
* Employing the GraphSAGE Graph Neural Network (GNN) architecture, combined with a link predictor, to leverage both these semantic embeddings and graph topology.
* Performing systematic training and evaluation using a robust link prediction pipeline, including hard negative sampling techniques for more challenging evaluation.
* Comparing the performance (AUC-ROC, Precision, Hits@K) across different embedding strategies.

The findings aim to demonstrate the value of incorporating deep semantic context from LLMs into graph mining tasks for improved link prediction.