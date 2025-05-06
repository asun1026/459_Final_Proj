import json
import logging
import os
import numpy as np
import time
import torch

# --- Configuration ---
TEXT_DATA_FILE = "DataExtract/Data/node_text_data.json" # Input: Map of node ID -> text
OUTPUT_FOLDER = "LLM_embed/embed_data" # Output folder

# --- Model Selection ---
MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
# Extract a short name for file naming
MODEL_SHORT_NAME = MODEL_NAME.split('/')[-1]

# --- GPU Selection ---
# Specify GPU indices (as strings) to make visible (e.g., ["0", "1"] or ["2", "3"]).
VISIBLE_GPU_INDICES = ["1", "3", "4"] 


# --- Embedding Parameters ---
# Which hidden layers to extract (0-based index from the hidden_states tuple)
# Layers 4, 8, 12, 16, 20, 24, 28 (every 4th, excluding last which is index 32)
# Hidden_states tuple includes input embeddings (idx 0) + output of each layer (idx 1-32)
TARGET_LAYER_INDICES = [4, 12, 20, 28]

# Maximum sequence length the model can handle (check model's config on Hugging Face)
# DeepSeek-R1 has a large context, but setting a reasonable limit is good practice
MAX_LENGTH = 1000 
BATCH_SIZE = 16 

MODEL_DTYPE = torch.bfloat16

# --- Set CUDA_VISIBLE_DEVICES Environment Variable ---
# This MUST be done *before* importing torch or transformers
if VISIBLE_GPU_INDICES:
    # Ensure indices are strings
    str_indices = [str(idx) for idx in VISIBLE_GPU_INDICES]
    cuda_visible_devices = ",".join(str_indices)
    print(f"INFO: Setting CUDA_VISIBLE_DEVICES to: '{cuda_visible_devices}'")
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
else:
    print("INFO: VISIBLE_GPU_INDICES not set. Using all GPUs visible to the system.")
# --- End CUDA_VISIBLE_DEVICES Setup ---

# --- Now import torch and transformers ---
# It's crucial these imports happen AFTER setting the environment variable
from transformers import AutoTokenizer, AutoModel # Or AutoModelForCausalLM
from tqdm.auto import tqdm # Progress bar

# --- Setup ---
# Configure logging after potential environment variable setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def load_json_file(filename):
    """Loads data from a JSON file."""
    # Add debug prints for path resolution
    current_working_directory = os.getcwd()
    logging.debug(f"DEBUG: Current Working Directory: {current_working_directory}")
    absolute_path_to_check = os.path.abspath(filename)
    logging.debug(f"DEBUG: Attempting to load from absolute path: {absolute_path_to_check}")
    logging.debug(f"DEBUG: Does the file exist at that path? {os.path.exists(absolute_path_to_check)}")

    if not os.path.exists(filename):
        # Use absolute path in error message for clarity
        logging.error(f"Error: File not found at resolved path: '{absolute_path_to_check}'. Relative path used: '{filename}'")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{filename}': {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading '{filename}': {e}")
        return None

def mean_pooling(model_output, attention_mask):
    """Performs mean pooling on token embeddings, ignoring padding tokens."""
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# --- Main Embedding Generation Script ---
def main():
    logging.info(f"Loading node text data from {TEXT_DATA_FILE}...")
    node_texts = load_json_file(TEXT_DATA_FILE)

    if node_texts is None:
        logging.error("Failed to load text data. Exiting.")
        return

    # Ensure keys are integers if they were loaded as strings from JSON
    try:
        node_texts_int_keys = {int(k): v for k, v in node_texts.items()}
        node_ids = list(node_texts_int_keys.keys())
        texts_to_encode = [node_texts_int_keys[nid] for nid in node_ids]
        logging.info(f"Loaded text for {len(node_ids)} nodes.")
    except (ValueError, TypeError) as e:
        logging.error(f"Error converting node IDs to integers: {e}. Make sure keys in {TEXT_DATA_FILE} are numeric strings.")
        return

    if not texts_to_encode:
        logging.warning("No text data found to encode.")
        return

    # --- Check CUDA availability (checks based on VISIBLE devices) ---
    if not torch.cuda.is_available():
        logging.error("CUDA not available according to PyTorch (check drivers and CUDA installation). This script requires GPU acceleration, especially with device_map='auto'. Exiting.")
        return

    # This count reflects only the *visible* GPUs after setting CUDA_VISIBLE_DEVICES
    num_visible_gpus = torch.cuda.device_count()
    logging.info(f"CUDA is available. Found {num_visible_gpus} VISIBLE GPU(s).")

    if num_visible_gpus == 0:
         logging.error("CUDA reports available, but no GPUs are visible after setting CUDA_VISIBLE_DEVICES (or none were set). Exiting.")
         return
    elif num_visible_gpus == 1:
         logging.warning("Only one GPU is visible/available. device_map='auto' will use only that GPU.")
    else:
         logging.info(f"Using device_map='auto' for distribution across {num_visible_gpus} visible GPUs.")

    # --- Load Model and Tokenizer ---
    logging.info(f"Loading Tokenizer and Model: {MODEL_NAME}...")
    # For gated models like Llama, ensure you are logged in via `huggingface-cli login`
    # or have HF_TOKEN environment variable set.
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Load the base model to get hidden states.
        # device_map='auto' requires the 'accelerate' library and will
        # automatically distribute across the GPUs made visible by CUDA_VISIBLE_DEVICES.
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True, # Crucial for getting intermediate layers
            device_map='auto', # Automatically distribute across VISIBLE GPUs
            torch_dtype=MODEL_DTYPE, # Use specified lower precision
            trust_remote_code=True # Needed for some models like Qwen
        )
        # No model.to(device) needed when device_map is used
        model.eval() # Set model to evaluation mode
        logging.info("Tokenizer and Model loaded successfully.")
        # This map shows how layers are assigned to the *visible* GPUs (indexed from 0)
        logging.info(f"Model device map (relative to visible devices): {model.hf_device_map}")
        # Get the primary device the model decided to use (often cuda:0 when using device_map)
        primary_device = model.device
        logging.info(f"Model's primary device (used for inputs): {primary_device}")

    except ImportError:
         logging.error("The 'accelerate' library is required for device_map='auto'. Please install it: pip install accelerate")
         return
    except Exception as e:
        logging.error(f"Failed to load Tokenizer or Model '{MODEL_NAME}': {e}")
        logging.error("Ensure model name is correct and you have internet connectivity.")
        logging.error("If using a gated model (like Llama), ensure you have provided access token via login or HF_TOKEN env var.")
        return

    # --- Prepare Storage for Embeddings from Multiple Layers ---
    # Dictionary to hold lists of batch embeddings for each target layer
    layer_batch_embeddings = {idx: [] for idx in TARGET_LAYER_INDICES}

    # --- Generate Embeddings ---
    logging.info(f"Generating embeddings for layers {TARGET_LAYER_INDICES} using mean pooling...")
    logging.info(f"Processing in batches of {BATCH_SIZE}...")
    start_time = time.time()
    output_device = 'cpu' # Where to collect final embeddings

    # Process in batches
    for i in tqdm(range(0, len(texts_to_encode), BATCH_SIZE), desc="Encoding Batches"):
        batch_texts = texts_to_encode[i:i + BATCH_SIZE]
        # Replace None or empty strings
        processed_batch = [text if text else " " for text in batch_texts]

        # Tokenize the batch - tokenizer usually runs on CPU
        inputs = tokenizer(
            processed_batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        # Explicitly move inputs to the model's primary device
        try:
            inputs = {k: v.to(primary_device) for k, v in inputs.items()}
        except Exception as move_err:
             logging.error(f"Error moving input batch {i//BATCH_SIZE} to device {primary_device}: {move_err}")
             continue # Skip this batch if moving fails

        # Get model outputs (no gradient calculation needed)
        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except Exception as forward_err:
                 logging.error(f"Error during model forward pass for batch {i//BATCH_SIZE}: {forward_err}", exc_info=True)
                 if "out of memory" in str(forward_err).lower():
                     logging.warning("Attempting to clear CUDA cache...")
                     torch.cuda.empty_cache()
                 continue # Skip this batch on error

        # Process each target layer
        try:
            # outputs.hidden_states is a tuple: (input_embed, layer1_out, layer2_out, ..., layerN_out)
            all_hidden_states = outputs.hidden_states
            if all_hidden_states is None or len(all_hidden_states) <= max(TARGET_LAYER_INDICES):
                 logging.error(f"Model did not return enough hidden states for batch {i//BATCH_SIZE}. Expected at least {max(TARGET_LAYER_INDICES)+1}, got {len(all_hidden_states) if all_hidden_states else 0}.")
                 continue

            attention_mask = inputs['attention_mask'] # Get attention mask once

            for layer_idx in TARGET_LAYER_INDICES:
                # Extract the specific layer's hidden state
                hidden_states = all_hidden_states[layer_idx]

                # Perform mean pooling
                # Ensure attention_mask is on the same device as hidden_states
                batch_embeddings = mean_pooling(hidden_states, attention_mask.to(hidden_states.device))

                # Move embeddings to CPU and store them associated with the layer index
                layer_batch_embeddings[layer_idx].append(batch_embeddings.to(output_device).numpy())

        except (AttributeError, IndexError, TypeError) as hs_err:
            logging.error(f"Could not extract/process hidden states for batch {i//BATCH_SIZE}: {hs_err}")
            continue # Skip batch if hidden state processing fails
        except Exception as pool_err:
            logging.error(f"Error during pooling for batch {i//BATCH_SIZE}: {pool_err}")
            continue # Skip batch if pooling fails


    end_time = time.time()
    logging.info(f"Batch processing complete. Time taken: {end_time - start_time:.2f} seconds.")

    # --- Concatenate and Save Embeddings for Each Layer ---
    # Create the output directory if it doesn't exist
    try:
        logging.info(f"Ensuring output directory exists: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directory {OUTPUT_FOLDER}: {e}")
        return

    logging.info("Concatenating and saving embeddings for each target layer...")
    node_ids_array = np.array(node_ids, dtype=int) # Prepare node IDs array once

    for layer_idx in TARGET_LAYER_INDICES:
        if not layer_batch_embeddings[layer_idx]:
            logging.warning(f"No embeddings were generated for layer {layer_idx}. Skipping save.")
            continue

        try:
            # Concatenate embeddings from all batches for this layer
            layer_embeddings_full = np.concatenate(layer_batch_embeddings[layer_idx], axis=0)
            logging.info(f"Layer {layer_idx}: Final embeddings shape: {layer_embeddings_full.shape}")

            # Define specific output filename for this layer
            # Use zfill to ensure consistent naming (e.g., layer_04, layer_08, layer_12)
            layer_output_filename = os.path.join(
                OUTPUT_FOLDER,
                f"node_hidden_state_embeddings_{MODEL_SHORT_NAME}_layer_{str(layer_idx).zfill(2)}.npz"
            )

            logging.info(f"Saving layer {layer_idx} embeddings to {layer_output_filename}...")
            np.savez_compressed(
                layer_output_filename,
                node_ids=node_ids_array,
                embeddings=layer_embeddings_full
            )
            logging.info(f"Layer {layer_idx} embeddings saved successfully.")

        except Exception as e:
            logging.error(f"Failed to concatenate or save embeddings for layer {layer_idx}: {e}")

    logging.info("All processing finished.")

if __name__ == "__main__":
    main()