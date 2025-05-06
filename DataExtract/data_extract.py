#!/usr/bin/env python
# -*- coding: utf-8 -*-
#DataExtract/data_extract.py

"""
Script to extract Stack Exchange network data from Common Crawl.
Uses cdx-toolkit to query the CDX index and requests to download WARC segments.
Filters out common block pages and JavaScript/cookie requirement pages.
Processes ALL candidate records found by the CDX query (up to INITIAL_CANDIDATE_LIMIT).
"""

import requests # For WARC download
import warcio
from warcio.archiveiterator import ArchiveIterator
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import math
import cdx_toolkit # For querying CDX index

# --- Configuration ---
# Single Crawl ID
CRAWL_ID = "CC-MAIN-2025-13" # Use the crawl ID you want to target

TARGET_DOMAINS = [
    "stackoverflow.com"
]

# Optional: Limit the total number of CANDIDATES queried from CDX
# Set to None to attempt to get *all* candidates found by cdx-toolkit.
# WARNING: Setting to None could result in millions of candidates, long run times,
# and very large output files. Set to a large number (e.g., 50000, 100000)
# if you want to cap the *input* size.
INITIAL_CANDIDATE_LIMIT = 14000 # Set to None or a large integer

OUTPUT_FOLDER = "DataExtract/Data"
# Output file to store results (JSON Lines format)
OUTPUT_FILE = os.join(OUTPUT_FOLDER, "stackexchange_cdxtoolkit_data_all_fixed.jsonl") # Adjusted output name
# Number of parallel workers for downloading/processing WARC records
MAX_WORKERS = 10 # Adjust based on your machine's capability and network

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processor_all_fixed.log"), # Log to file
        logging.StreamHandler() # Also log to console
    ]
)

# --- Helper Functions ---
# is_target_domain function remains the same
def is_target_domain(url):
    if not url: return False
    try:
        hostname = urlparse(url).hostname
        if hostname:
            hostname = hostname.lower()
            if hostname.startswith("www."): hostname = hostname[4:]
            for domain in TARGET_DOMAINS:
                if domain.startswith("."):
                    if hostname.endswith(domain): return True
                else:
                    if hostname == domain: return True
    except Exception as e:
         logging.debug(f"Error parsing URL {url} in is_target_domain: {e}")
    return False

# --- CDX Index Query Function (using cdx-toolkit - FIXED) ---
def query_cdx_index(crawl_id, domains, limit=None):
    """
    Queries the Common Crawl CDX index using cdx-toolkit, retrieving
    up to 'limit' candidate records matching the domains for the specified crawl.
    Handles limit=None correctly.
    """
    logging.info(f"Querying CDX using cdx-toolkit for crawl: {crawl_id}")
    try:
        cdx = cdx_toolkit.CDXFetcher(source='cc')
    except Exception as e:
        logging.error(f"Failed to initialize CDXFetcher: {e}")
        return []

    records = []
    processed_urls = set()
    limit_reached = False # Flag to stop processing if limit hit

    # Loop through each target domain pattern
    for domain_pattern in domains:
        if limit_reached: break # Stop querying new patterns if limit already hit

        # Construct query pattern
        if domain_pattern.startswith("."):
            query_url_pattern = "*" + domain_pattern
        else:
            query_url_pattern = domain_pattern + "/*"

        logging.info(f"--- Querying cdx-toolkit for pattern: {query_url_pattern} (Candidate Limit: {limit}) ---")

        try:
            # --- !!! FIX 1: Handle limit=None !!! ---
            # Prepare parameters for cdx.iter()
            iter_params = {
                'url': query_url_pattern,
                'cc_indexes': [crawl_id],
                'filter': ['=mime:text/html']
                # Add other filters like '=status:200' here if desired
            }
            # Only add the limit parameter if it's not None
            if limit is not None:
                iter_params['limit'] = limit

            # Call cdx.iter with keyword arguments
            stream = cdx.iter(**iter_params)
            # --- End FIX 1 ---

            processed_count_pattern = 0
            # Iterate through the results yielded by cdx-toolkit
            for record_obj in stream:
                record = None
                # Handle different potential return types
                if isinstance(record_obj, dict): record = record_obj
                elif hasattr(record_obj, 'data') and isinstance(record_obj.data, dict): record = record_obj.data
                elif isinstance(record_obj, str):
                    try: record = json.loads(record_obj)
                    except (json.JSONDecodeError, TypeError):
                        logging.warning(f"Skipping non-dict/JSON record from cdx.iter: {record_obj[:100]}...")
                        continue
                else:
                    logging.warning(f"Skipping unexpected record type from cdx.iter: {type(record_obj)}")
                    continue

                url = record.get('url')

                # Validation Checks
                if (url and
                    is_target_domain(url) and
                    url not in processed_urls and
                    all(k in record for k in ('filename', 'offset', 'length'))):
                    try:
                        # Ensure offset/length are integers
                        record['offset'] = int(record['offset'])
                        record['length'] = int(record['length'])
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Skipping record with invalid offset/length for {url}: {e}")
                        continue

                    records.append(record)
                    processed_urls.add(url)
                    processed_count_pattern += 1

                    # Check if the overall CANDIDATE limit has been reached
                    # This check might be slightly redundant if cdx.iter(limit=...) works perfectly,
                    # but provides an extra layer of control.
                    if limit and len(records) >= limit:
                        logging.info(f"Reached overall candidate query limit of {limit} during iteration.")
                        limit_reached = True
                        break # Stop processing records from this stream

            logging.info(f"Processed {processed_count_pattern} unique records for pattern {query_url_pattern}")

        # --- !!! FIX 2: Removed incorrect except block !!! ---
        # Removed: except cdx_toolkit.CDXNotFound:
        except Exception as e:
            # Catch potential errors from cdx-toolkit or iteration
            logging.error(f"Error querying pattern {query_url_pattern} with cdx-toolkit: {e}", exc_info=True)
            # Continue to the next domain pattern

    logging.info(f"Total relevant candidate records collected via cdx-toolkit: {len(records)}")
    return records


# --- WARC Record Processing Function (using requests - CORRECTED + ENHANCED FILTERING) ---
def process_warc_record(record_info):
    """Downloads WARC segment, extracts text/links, filters bad pages."""
    filename = record_info.get('filename')
    # Ensure offset/length are integers from the start
    try:
        offset = int(record_info.get('offset'))
        length = int(record_info.get('length'))
    except (ValueError, TypeError, AttributeError):
        logging.warning(f"Invalid offset/length in record_info: {record_info}. Skipping.")
        return None

    # --- !!! THIS LINE WAS MISSING - ADDED BACK !!! ---
    cdx_page_url = record_info.get('url')
    # --- End Added Line ---

    # Basic validation of essential info
    if not all([filename, isinstance(offset, int), isinstance(length, int), length > 0, cdx_page_url]):
         logging.warning(f"Skipping record due to missing/invalid info: {record_info}")
         return None

    warc_file_url = f"https://data.commoncrawl.org/{filename}"
    start_byte = offset
    end_byte = offset + length - 1
    byte_range = f'bytes={start_byte}-{end_byte}'
    headers = {'Range': byte_range}
    # Initialize default return structure, use CDX URL initially
    # Now cdx_page_url is defined before use here
    extracted_data = {'url': cdx_page_url, 'text': None, 'links': []}

    try:
        with requests.get(warc_file_url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
            response.raw.decode_content = True # Handle potential gzip

            stream = ArchiveIterator(response.raw)
            processed_record_in_stream = False # Flag
            for record in stream:
                # Check HTTP Status Code from WARC record headers
                status_code = record.http_headers.get_statuscode() if record.http_headers else None
                if status_code != '200':
                    logging.debug(f"Skipping record for {cdx_page_url} due to HTTP status: {status_code}")
                    continue # Skip non-200 responses within the WARC

                # Check Content-Type
                if record.rec_type == 'response' and record.http_headers.get_header('Content-Type', '').startswith('text/html'):
                    processed_record_in_stream = True # Found relevant record
                    record_url = record.rec_headers.get_header('WARC-Target-URI')
                    page_url_for_links = cdx_page_url # Default
                    if record_url:
                        extracted_data['url'] = record_url # Update URL
                        page_url_for_links = record_url # Use for link resolution

                    html_content_bytes = record.content_stream().read()
                    # Decoding
                    try:
                        html_content = html_content_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            html_content = html_content_bytes.decode('iso-8859-1', errors='ignore')
                            logging.debug(f"Decoding issue for {page_url_for_links}, used iso-8859-1 fallback.")
                        except Exception as decode_err:
                            logging.error(f"Failed to decode content for {page_url_for_links}: {decode_err}")
                            return None # Treat as failure

                    if not html_content:
                         logging.warning(f"Empty HTML content after decoding for {page_url_for_links}. Skipping.")
                         return None # Treat as failure

                    # Text Extraction (Trafilatura)
                    try:
                        extracted_text = trafilatura.extract(html_content, include_comments=False, include_tables=False, no_fallback=True)
                        if extracted_text:
                            cleaned_text = ' '.join(extracted_text.split())
                            cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
                            extracted_data['text'] = cleaned_text
                        else:
                            extracted_data['text'] = ""
                    except Exception as te:
                         logging.error(f"Trafilatura failed for {page_url_for_links}: {te}")
                         extracted_data['text'] = "" # Assign empty text

                    # Enhanced Filtering (Block Pages, JS Walls, Boilerplate, Short Text)
                    filter_page = False
                    filter_reason = ""
                    if extracted_data['text'] is not None:
                        text_lower = extracted_data['text'].lower()
                        if ("has been blocked from access" in text_lower or
                            "cdn-cgi/l/email-protection" in text_lower or
                            "ray id:" in text_lower or
                            "enable javascript and cookies to continue" in text_lower):
                            filter_page = True
                            filter_reason = "Block/JS Page"
                        elif ("communities for your favorite technologies" in text_lower and
                              "explore all collectives" in text_lower):
                            filter_page = True
                            filter_reason = "Boilerplate (Collectives/Teams)"
                        elif ("page not found" in text_lower and
                              "couldn't find the page you requested" in text_lower):
                             filter_page = True
                             filter_reason = "404 Page Text"
                        # Add more specific patterns if needed

                    if not filter_page and (extracted_data['text'] is None or len(extracted_data['text']) < 50): # Arbitrary short length check
                        filter_page = True
                        filter_reason = f"Very Short Text ({len(extracted_data.get('text', ''))} chars)"

                    if filter_page:
                        logging.debug(f"Filtering out {extracted_data['url']} due to: {filter_reason}")
                        return None # Discard this record

                    # Link Extraction (BeautifulSoup)
                    try:
                        soup = BeautifulSoup(html_content, 'lxml')
                        links = []
                        for a_tag in soup.find_all('a', href=True):
                            href = a_tag.get('href')
                            if not href: continue
                            try:
                                absolute_link = urljoin(page_url_for_links, href.strip())
                                if absolute_link.startswith(('http://', 'https://')) and is_target_domain(absolute_link):
                                    links.append(absolute_link)
                            except ValueError:
                                logging.debug(f"Could not resolve relative url '{href}' for base '{page_url_for_links}'")
                        extracted_data['links'] = links
                    except Exception as bs_e:
                        logging.error(f"BeautifulSoup link extraction failed for {page_url_for_links}: {bs_e}")
                        extracted_data['links'] = [] # Set empty links on failure

                    # If we passed all filters and extraction steps for this record
                    return extracted_data

            # After iterating through stream: Check if we processed the expected record type
            if not processed_record_in_stream:
                 logging.debug(f"HTML response record (Status 200) not found in downloaded WARC segment for {cdx_page_url} at {warc_file_url} range {byte_range}")
            return None # Return None if no suitable record found in stream

    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        # Log only actual errors, not expected skips like 404s unless debugging
        if status_code != 404: # Example: Don't spam logs with 404s during download
             logging.warning(f"HTTP request failed for {cdx_page_url} (Status: {status_code}) - Skipping. Error: {e}")
        else:
             logging.debug(f"HTTP request failed for {cdx_page_url} (Status: {status_code}) - Skipping.")
        return None
    except Exception as e:
         # Catch any other unexpected errors during processing (warcio parsing, etc.)
         logging.error(f"Unexpected error in process_warc_record for {cdx_page_url} (File: {filename}): {e}", exc_info=False) # exc_info=False to reduce log spam
         return None

# --- Main Execution Logic (No success limit) ---
if __name__ == "__main__":
    logging.info(f"Starting SE data extraction from crawl {CRAWL_ID} using cdx-toolkit...")
    if INITIAL_CANDIDATE_LIMIT:
        logging.info(f"Will query CDX for up to {INITIAL_CANDIDATE_LIMIT} candidate records.")
    else:
        logging.info("Will query CDX for ALL available candidate records (no limit).")

    # 1. Query CDX Index using the cdx-toolkit function
    cdx_records = query_cdx_index(CRAWL_ID, TARGET_DOMAINS, limit=INITIAL_CANDIDATE_LIMIT)

    num_candidates = len(cdx_records)
    if not cdx_records:
        logging.warning("No candidate records found using cdx-toolkit. Exiting.")
        exit()

    logging.info(f"Total unique candidates found via cdx-toolkit: {num_candidates}")
    logging.info(f"Processing all {num_candidates} candidates...")

    # 2. Process Records in Parallel
    processed_count = 0
    failed_filtered_count = 0
    tasks_submitted = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:

        future_to_record = {executor.submit(process_warc_record, record): record for record in cdx_records}
        tasks_submitted = len(future_to_record)
        logging.info(f"Submitted {tasks_submitted} processing tasks to executor.")

        # Process ALL completed tasks
        for future in as_completed(future_to_record):
            record_info = future_to_record[future]
            try:
                result = future.result()
                if result and result.get('text') is not None: # Success and not filtered
                    outfile.write(json.dumps(result) + '\n')
                    processed_count += 1
                else: # Failed or filtered
                     failed_filtered_count += 1

                # Log progress periodically
                total_completed = processed_count + failed_filtered_count
                if total_completed % 100 == 0 and total_completed > 0: # Log every 100 completed tasks
                    elapsed = time.time() - start_time
                    rate = total_completed / elapsed if elapsed > 0 else 0
                    logging.info(f"Progress: {total_completed}/{tasks_submitted} tasks completed. "
                                 f"Success: {processed_count}, Failed/Filtered: {failed_filtered_count}. Rate: {rate:.2f} tasks/sec.")

                # --- NO BREAK CONDITION HERE ---

            except Exception as e:
                retrieval_url = record_info.get('url', 'unknown URL')
                logging.error(f"An exception occurred retrieving result for task associated with {retrieval_url}: {e}", exc_info=True)
                failed_filtered_count += 1

    # Final Summary
    end_time = time.time()
    total_completed = processed_count + failed_filtered_count
    logging.info(f"--- Processing Complete ---")
    logging.info(f"Total time: {end_time - start_time:.2f} seconds")
    logging.info(f"Total tasks completed: {total_completed} out of {tasks_submitted} submitted.")
    logging.info(f"Successfully processed and saved: {processed_count} pages")
    logging.info(f"Failed or Filtered (block pages, errors, etc.): {failed_filtered_count} pages")
    logging.info(f"Results saved to: {OUTPUT_FILE}")

    # Optional: Log the success rate
    if tasks_submitted > 0:
        success_rate = (processed_count / tasks_submitted) * 100
        logging.info(f"Overall success rate (Saved / Submitted): {success_rate:.2f}%")