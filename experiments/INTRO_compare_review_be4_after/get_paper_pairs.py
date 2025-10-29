import os
import csv
import ast
import shutil
import datetime
from io import StringIO

from tqdm import tqdm

# --- Configuration ---

# The main CSV file with paper information
CSV_FILE = 'ICLR2024.csv'

# Flawed papers CSV file with flaw descriptions
FLAWED_PAPERS_CSV = os.path.join('../../data', 'flawed_papers', 'ICLR2024_latest_flawed_papers_v1', 'flawed_papers_global_summary.csv')

# Base directories for the parsed paper versions
# Based on your screenshot, the data is in a 'data' folder
V1_DIR = os.path.join('data', 'ICLR2024_v1')
LATEST_DIR = os.path.join('data', 'ICLR2024_latest')

# Directory to store the output pairs
OUTPUT_DIR = os.path.join('data', 'ICLR2024_pairs')

# Date boundaries (as UTC-aware datetime objects)
BOUNDARY_DATE = datetime.datetime(2024, 1, 15, 0, 0, 0, tzinfo=datetime.timezone.utc)
END_DATE = datetime.datetime(2025, 6, 25, 0, 0, 0, tzinfo=datetime.timezone.utc)

# --- End Configuration ---


def load_flaw_descriptions(flawed_papers_csv_path):
    """
    Loads flaw descriptions from the flawed papers CSV and groups them by paper ID.
    Returns a dictionary mapping paper_id -> list of flaw descriptions.
    """
    flaw_dict = {}
    
    if not os.path.exists(flawed_papers_csv_path):
        print(f"Warning: Flawed papers CSV not found at {flawed_papers_csv_path}")
        return flaw_dict
    
    try:
        # Read file and filter out NUL characters
        with open(flawed_papers_csv_path, mode='r', encoding='utf-8', errors='ignore') as f:
            content = f.read().replace('\x00', '')
        
        # Parse the cleaned content
        reader = csv.DictReader(StringIO(content))
        for row in reader:
            openreview_id = row.get('openreview_id')
            flaw_description = row.get('flaw_description')
            
            if openreview_id and flaw_description:
                if openreview_id not in flaw_dict:
                    flaw_dict[openreview_id] = []
                flaw_dict[openreview_id].append(flaw_description)
        
        # print(f"Loaded flaw descriptions for {len(flaw_dict)} papers")
    except Exception as e:
        # print(f"Warning: Could not load flawed papers CSV: {e}")
        pass
    
    return flaw_dict


def parse_arxiv_date(date_str):
    """
    Converts the ISO format string from the CSV (with 'Z')
    into a UTC-aware datetime object.
    """
    try:
        # Replace 'Z' with '+00:00' for Python's fromisoformat
        return datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None

def find_paper_folder(base_dir, folder_name):
    """
    Searches for a folder in the 'accepted' and 'rejected'
    subdirectories of the given base_dir.
    """
    # search_dirs = [os.path.join(base_dir, 'accepted'), os.path.join(base_dir, 'rejected')]
    search_dirs = [os.path.join(base_dir, 'accepted')]
    for subdir in search_dirs:
        target_path = os.path.join(subdir, folder_name)
        if os.path.isdir(target_path) and os.path.exists(os.path.join(target_path, 'structured_paper_output', 'paper.md')):
            return target_path
    return None

def get_latest_version(arxiv_info):
    """
    Finds the latest version key (e.g., 'v2', 'v3') from the arxiv_info dict.
    Returns (latest_version_key, latest_date_str) or (None, None).
    """
    if not isinstance(arxiv_info, dict) or not arxiv_info:
        return None, None

    # Sort keys by version number (e.g., 'v1', 'v2', 'v10')
    try:
        sorted_versions = sorted(
            arxiv_info.keys(),
            key=lambda v: int(v[1:]) if v.startswith('v') and v[1:].isdigit() else -1
        )
    except ValueError:
        # Handle non-standard keys if any
        return None, None

    if not sorted_versions:
        return None, None

    latest_key = sorted_versions[-1]
    return latest_key, arxiv_info[latest_key]


def main():
    print(f"Starting paper pair processing...")
    print(f"Source CSV: {CSV_FILE}")
    print(f"Flawed Papers CSV: {FLAWED_PAPERS_CSV}")
    print(f"V1 Directory: {V1_DIR}")
    print(f"Latest Directory: {LATEST_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}\n")

    # Load flaw descriptions
    flaw_descriptions = load_flaw_descriptions(FLAWED_PAPERS_CSV)
    print(f"Loaded flaw descriptions for {len(flaw_descriptions)} papers\n")

    # Create output directories
    output_v1_dir = os.path.join(OUTPUT_DIR, 'v1')
    output_latest_dir = os.path.join(OUTPUT_DIR, 'latest')
    os.makedirs(output_v1_dir, exist_ok=True)
    os.makedirs(output_latest_dir, exist_ok=True)

    filtered_pairs = []
    processed_count = 0
    found_count = 0

    if not os.path.exists(CSV_FILE):
        print(f"ERROR: Cannot find source CSV file: {CSV_FILE}")
        return

    try:
        # First, count the total number of rows for progress bar
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            total_rows = sum(1 for line in f) - 1  # Subtract 1 for header
        
        # Now process the rows
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in tqdm(reader, total=total_rows, desc="Processing rows"):
                processed_count += 1
                paper_id = row.get('paperid')
                arxiv_id = row.get('arxiv_id')
                arxiv_info_str = row.get('arxiv_info')

                # Skip rows without necessary info
                if not all([paper_id, arxiv_id, arxiv_info_str]):
                    continue

                # Parse the arxiv_info string into a dict
                try:
                    arxiv_info = ast.literal_eval(arxiv_info_str)
                    if not isinstance(arxiv_info, dict) or 'v1' not in arxiv_info:
                        continue
                except (ValueError, SyntaxError):
                    # print(f"Warning: Could not parse arxiv_info for {paper_id}")
                    continue

                # 1. Check v1 date
                v1_date = parse_arxiv_date(arxiv_info.get('v1'))
                if not v1_date or v1_date >= BOUNDARY_DATE:
                    continue # v1 is not before the boundary date

                # 2. Find and check latest version date
                latest_key, latest_date_str = get_latest_version(arxiv_info)
                if not latest_key or latest_key == 'v1':
                    continue # No version newer than v1

                latest_date = parse_arxiv_date(latest_date_str)
                if not latest_date:
                    continue

                # 3. Apply date filters
                if (latest_date > BOUNDARY_DATE and latest_date < END_DATE):
                    # This is a valid pair!
                    # print(f"\nFound potential pair for paperid: {paper_id} (Arxiv: {arxiv_id})")
                    # print(f"  v1: {v1_date.date()} (valid)")
                    # print(f"  {latest_key}: {latest_date.date()} (valid)")

                    # Construct folder names to search for
                    formatted_arxiv_id = arxiv_id.replace('.', '_')
                    v1_folder_name = f"{paper_id}_{formatted_arxiv_id}v1"
                    latest_folder_name = f"{paper_id}_{formatted_arxiv_id}" #{latest_key}"

                    # 4. Find and copy folders
                    source_v1_path = find_paper_folder(V1_DIR, v1_folder_name)
                    source_latest_path = find_paper_folder(LATEST_DIR, latest_folder_name)

                    if source_v1_path and source_latest_path:
                        # print(f"  > Found v1 folder: {source_v1_path}")
                        # print(f"  > Found latest folder: {source_latest_path}")

                        # Copy folders to the output directory
                        dest_v1_path = os.path.join(output_v1_dir, v1_folder_name)
                        dest_latest_path = os.path.join(output_latest_dir, latest_folder_name)

                        try:
                            if not os.path.exists(dest_v1_path):
                                shutil.copytree(source_v1_path, dest_v1_path)
                            if not os.path.exists(dest_latest_path):
                                shutil.copytree(source_latest_path, dest_latest_path)

                            # Add to list for final CSV
                            pair_info = row.copy()
                            pair_info['v1_folder_path'] = dest_v1_path
                            pair_info['latest_folder_path'] = dest_latest_path
                            pair_info['latest_version_key'] = latest_key
                            
                            # Add flaw descriptions as a list
                            if paper_id in flaw_descriptions:
                                pair_info['flaw_descriptions'] = flaw_descriptions[paper_id]
                                # print(f"  > Added {len(flaw_descriptions[paper_id])} flaw description(s)")
                            else:
                                pair_info['flaw_descriptions'] = []
                                # print(f"  > No flaw descriptions found for this paper")
                            
                            filtered_pairs.append(pair_info)
                            found_count += 1

                        except (shutil.Error, OSError) as e:
                            print(f"  ! ERROR copying files for {paper_id}: {e}")

                    # else:
                        # if not source_v1_path:
                            # print(f"  ! Warning: Could not find v1 folder: {v1_folder_name}")
                        # if not source_latest_path:
                            # print(f"  ! Warning: Could not find latest folder: {latest_folder_name}")

    except FileNotFoundError:
        print(f"ERROR: Source CSV file not found at {CSV_FILE}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # 5. Write the new CSV with all filtered pair info
    if filtered_pairs:
        output_csv_path = os.path.join(OUTPUT_DIR, 'filtered_pairs.csv')
        print(f"\nWriting {len(filtered_pairs)} found pairs to {output_csv_path}...")

        # Dynamically get fieldnames from the first record
        fieldnames = filtered_pairs[0].keys()
        
        try:
            with open(output_csv_path, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_pairs)
        except IOError as e:
            print(f"ERROR: Could not write output CSV: {e}")

    print("\n--- Processing Complete ---")
    print(f"Total rows scanned: {processed_count}")
    print(f"Total pairs found and copied: {found_count}")
    print(f"Output data is in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
