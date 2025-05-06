# DataExtract/graph_builder.py
import json
import os
import networkx as nx
import logging
from urllib.parse import urlparse, urldefrag # For basic canonicalization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_FOLDER = "DataExtract/Data"

INPUT_JSONL_FILE = "Data/stackexchange_cdxtoolkit_data_all_fixed.jsonl"
GRAPH_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "stackexchange_graph.gexf") # Or .adjlist, .pkl etc.

def canonicalize_url(url):
    """Basic URL canonicalization: remove fragment, add scheme if missing."""
    try:
        # Remove fragment (#...)
        url_defrag, _ = urldefrag(url)
        parsed = urlparse(url_defrag)
        # Add scheme if missing (assuming https for SE sites)
        scheme = parsed.scheme or 'https'
        # Reconstruct (simple version, ignores params/query sorting for now)
        # Lowercase hostname
        netloc = parsed.netloc.lower()
        # Remove www. if present? Optional, decide on a standard. Let's keep it for now.
        # Example: parsed.path is case-sensitive, keep it as is.
        canon_url = f"{scheme}://{netloc}{parsed.path}"
        if parsed.query:
             canon_url += f"?{parsed.query}" # Keep query for now, might need smarter handling
        return canon_url
    except Exception as e:
        logging.warning(f"Could not canonicalize URL {url}: {e}")
        return None

def build_graph_from_jsonl(jsonl_path):
    """Reads the JSONL file and builds a directed graph using NetworkX."""
    logging.info(f"Starting graph construction from {jsonl_path}")
    graph = nx.DiGraph()
    nodes_data = {} # Store URL -> {'text': ..., 'links': [...]}

    # --- First pass: Read data and identify all potential nodes ---
    logging.info("First pass: Reading data and identifying nodes...")
    node_urls = set()
    processed_lines = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                url = data.get('url')
                text = data.get('text')
                links = data.get('links', [])

                if not url or text is None: # Skip if no URL or text is missing
                    logging.warning(f"Skipping line due to missing URL or text: {line.strip()}")
                    continue

                canon_url = canonicalize_url(url)
                if canon_url:
                    node_urls.add(canon_url)
                    # Store data associated with the canonical URL
                    # If duplicate canonical URLs exist (rare), last one wins here
                    nodes_data[canon_url] = {'text': text, 'links': links}
                processed_lines += 1
                if processed_lines % 10000 == 0:
                     logging.info(f"Processed {processed_lines} lines for node identification...")

            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                 logging.error(f"Error processing line: {line.strip()} - Error: {e}")

    logging.info(f"Identified {len(node_urls)} unique canonical node URLs.")

    # Add nodes to the graph
    # Assign integer IDs for potentially easier handling later
    url_to_id = {url: i for i, url in enumerate(node_urls)}
    id_to_url = {i: url for url, i in url_to_id.items()}
    graph.add_nodes_from(id_to_url.keys()) # Add nodes using integer IDs
    logging.info(f"Added {graph.number_of_nodes()} nodes to the graph.")

    # --- Second pass: Add edges ---
    logging.info("Second pass: Adding edges...")
    edge_count = 0
    processed_nodes_for_edges = 0
    for source_canon_url, data in nodes_data.items():
        source_node_id = url_to_id.get(source_canon_url)
        if source_node_id is None: continue # Should not happen if logic is correct

        for target_raw_link in data['links']:
            target_canon_url = canonicalize_url(target_raw_link)
            if target_canon_url and target_canon_url in url_to_id:
                target_node_id = url_to_id[target_canon_url]
                if source_node_id != target_node_id: # Avoid self-loops unless desired
                    if not graph.has_edge(source_node_id, target_node_id):
                         graph.add_edge(source_node_id, target_node_id)
                         edge_count += 1

        processed_nodes_for_edges += 1
        if processed_nodes_for_edges % 10000 == 0:
            logging.info(f"Processed {processed_nodes_for_edges} nodes for edge creation...")


    logging.info(f"Added {edge_count} edges to the graph.")
    logging.info(f"Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    return graph, url_to_id, id_to_url, nodes_data

# --- Build and Save Graph ---
if __name__ == "__main__":
    graph, url_to_id, id_to_url, nodes_data = build_graph_from_jsonl(INPUT_JSONL_FILE)

    # You might want to save the graph, mappings, and text data separately
    logging.info(f"Saving graph to {GRAPH_OUTPUT_FILE}...")
    nx.write_gexf(graph, GRAPH_OUTPUT_FILE) # GEXF format preserves node IDs well
    logging.info("Graph saved.")

    # Save mappings and text data (optional, but useful)
    MAPPING_FILE = os.path.join(OUTPUT_FOLDER,"url_id_mapping.json")
    TEXT_DATA_FILE = os.path.join(OUTPUT_FOLDER,"node_text_data.json") # Map ID -> text

    logging.info(f"Saving URL-ID mappings to {MAPPING_FILE}...")
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump({'url_to_id': url_to_id, 'id_to_url': id_to_url}, f)
    logging.info("Mappings saved.")

    logging.info(f"Saving Node ID -> Text mapping to {TEXT_DATA_FILE}...")
    node_texts = {url_to_id[url]: data['text'] for url, data in nodes_data.items() if url in url_to_id}
    with open(TEXT_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(node_texts, f) # Saves as {"id": "text", ...}
    logging.info("Text data saved.")