import json
import os
import pandas as pd
from collections import defaultdict
import re
import shutil

def load_json_file(filepath):
    """Safely loads a single JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read or parse {filepath}: {e}")
        return None

def collect_metareview_data(metareviews_path, venue_folder_name):
    """
    Collects 'is_flaw_mentioned' and 'mention_reasoning' from metareview JSON files.
    """
    mention_data = {}
    print(f"Scanning for metareview data in: {metareviews_path}")

    if not os.path.exists(metareviews_path):
        print(f"Warning: Metareviews directory not found at {metareviews_path}")
        return mention_data

    for model_name in ['o3']:# os.listdir(metareviews_path):
        model_path = os.path.join(metareviews_path, model_name)
        if os.path.isdir(model_path):
            venue_path = os.path.join(model_path, venue_folder_name)
            if os.path.isdir(venue_path):
                for status in ['accepted', 'rejected']:
                    status_path = os.path.join(venue_path, status)
                    if os.path.isdir(status_path):
                        for filename in os.listdir(status_path):
                            if filename.endswith(".json"):
                                data = load_json_file(os.path.join(status_path, filename))
                                if not data:
                                    continue
                                for paper_key, flaws in data.items():
                                    openreview_id = paper_key.split('_')[0]
                                    for flaw in flaws:
                                        flaw_id = flaw.get('flaw_id')
                                        if openreview_id and flaw_id:
                                            key = (openreview_id, flaw_id)
                                            mention_data[key] = {
                                                'is_flaw_mentioned': flaw.get('is_flaw_mentioned'),
                                                'mention_reasoning': flaw.get('mention_reasoning')
                                            }
    print(f"Collected mention data for {len(mention_data)} flaws.")
    return mention_data

def collect_llm_review_data(reviews_path, venue_folder_name):
    """
    Collects the full LLM review content from individual review JSON files.
    """
    review_data = {}
    print(f"Scanning for LLM review data in: {reviews_path}")

    if not os.path.exists(reviews_path):
        print(f"Warning: Reviews directory not found at {reviews_path}")
        return review_data

    for model_name in ['o3']: #os.listdir(reviews_path):
        model_path = os.path.join(reviews_path, model_name)
        if os.path.isdir(model_path):
            venue_path = os.path.join(model_path, venue_folder_name)
            if os.path.isdir(venue_path):
                for status in ['accepted', 'rejected']:
                    status_path = os.path.join(venue_path, status)
                    if not os.path.isdir(status_path): continue
                    for paper_folder in os.listdir(status_path):
                        paper_folder_path = os.path.join(status_path, paper_folder)
                        if not os.path.isdir(paper_folder_path): continue
                        
                        openreview_id = paper_folder.split('_')[0]
                        for filename in os.listdir(paper_folder_path):
                            if filename.endswith("_review.json"):
                                # Extract flaw_id from filename
                                # e.g., 0aN7VWwp4g_2410_23159_incorrect_csi_thresholds_review.json
                                # -> incorrect_csi_thresholds
                                match = re.match(r'(.+?)_(\d+_\d+)_(.+)_review\.json', filename)
                                if match:
                                    flaw_id = match.group(3)
                                else:
                                    # Fallback for different naming
                                    base_name = filename.replace('_review.json', '')
                                    # Assuming the last part is the flaw id
                                    flaw_id = '_'.join(base_name.split('_')[3:])

                                if not flaw_id: continue
                                
                                data = load_json_file(os.path.join(paper_folder_path, filename))
                                if data is not None:
                                    key = (openreview_id, flaw_id)
                                    review_data[key] = {'llm_review': json.dumps(data, indent=2)}
    
    print(f"Collected LLM reviews for {len(review_data)} flaws.")
    return review_data

def create_aggregated_dataset(venue_folder_name, base_data_dir, categorized_data_dir, output_filename):
    """
    Main function to orchestrate the data aggregation process.
    """
    # Define paths to the different data sources
    flawed_papers_dir = os.path.join(base_data_dir, 'flawed_papers', venue_folder_name)
    metareviews_dir = os.path.join(base_data_dir, 'metareviews')
    reviews_dir = os.path.join(base_data_dir, 'reviews')

    # --- 1. Load Base Data: Categories and Descriptions ---
    print("Step 1: Loading base data...")
    # Load categorized flaws (openreview_id, flaw_id, category_ids)
    categories_path = os.path.join(categorized_data_dir, 'flawed_papers', venue_folder_name, 'categorized_flaw_cleaned.csv')
    try:
        categories_df = pd.read_csv(categories_path)
    except FileNotFoundError:
        print(f"Error: Base category file not found at {categories_path}. Cannot proceed.")
        return

    # Load flaw descriptions
    descriptions_path = os.path.join(flawed_papers_dir, 'flawed_papers_global_summary.csv')
    try:
        descriptions_df = pd.read_csv(descriptions_path)[['openreview_id', 'flaw_id', 'flaw_description', 'llm_generated_modifications']]
    except FileNotFoundError:
        print(f"Warning: Flaw description file not found at {descriptions_path}.")
        descriptions_df = pd.DataFrame(columns=['openreview_id', 'flaw_id', 'flaw_description', 'llm_generated_modifications'])

    # Merge categories and descriptions
    df = pd.merge(categories_df, descriptions_df, on=['openreview_id', 'flaw_id'], how='left')
    print(f"Loaded and merged base data. Shape: {df.shape}")

    # --- 2. Collect and Merge Metareview Data ---
    print("\nStep 2: Collecting metareview data...")
    mention_data = collect_metareview_data(metareviews_dir, venue_folder_name)
    mention_df = pd.DataFrame.from_dict(mention_data, orient='index').reset_index()
    mention_df.rename(columns={'level_0': 'openreview_id', 'level_1': 'flaw_id'}, inplace=True)
    df = pd.merge(df, mention_df, on=['openreview_id', 'flaw_id'], how='left')
    print(f"Data shape after merging mention data: {df.shape}")

    # --- 3. Collect and Merge LLM Review Data ---
    print("\nStep 3: Collecting LLM review data...")
    review_data = collect_llm_review_data(reviews_dir, venue_folder_name)
    review_df = pd.DataFrame.from_dict(review_data, orient='index').reset_index()
    review_df.rename(columns={'level_0': 'openreview_id', 'level_1': 'flaw_id'}, inplace=True)
    df = pd.merge(df, review_df, on=['openreview_id', 'flaw_id'], how='left')
    print(f"Data shape after merging LLM review data: {df.shape}")
    
    # --- 4. Finalize and Save ---
    print("\nStep 4: Finalizing and saving the dataset...")
    # Ensure all required columns are present
    final_columns = [
        'openreview_id', 'flaw_id', 'category_ids', 'flaw_description', 'llm_generated_modifications',
        'llm_review', 'is_flaw_mentioned', 'mention_reasoning'
    ]
    for col in final_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns and save to CSV
    final_df = df[final_columns]
    final_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully created aggregated dataset with {len(final_df)} rows.")
    print(f"File saved to: {output_filename}")

def sample_flaw_data(input_csv_path, category_ids, n_samples_per_group=10, require_llm_review=False, require_metareview=False, seed=42):
    """
    Samples data from the aggregated dataset for a list of categories.
    It ensures each flaw is sampled at most once and attempts to collect
    n_samples_per_group for both 'mentioned' and 'not-mentioned' groups
    for each category from the available pool of unique flaws.

    Args:
        input_csv_path (str): The path to the aggregated CSV file.
        category_ids (list): A list of flaw category IDs to filter by.
        n_samples_per_group (int): The number of samples to draw per group.
        require_llm_review (bool): If True, only samples from flaws that have an LLM review.
        require_metareview (bool): If True, only samples from flaws that have metareview data.
        seed (int): The random seed for reproducibility.
        
    Returns:
        pd.DataFrame: A DataFrame containing the balanced and unique sample.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return None

    

    # Create a unique identifier for each flaw to make filtering easier and faster
    df['unique_flaw_id'] = df['openreview_id'].astype(str) + "||" + df['flaw_id'].astype(str)
    
    # Set to keep track of unique_flaw_ids that have already been sampled
    sampled_flaw_ids = set()
    
    all_samples_list = []

    for category_id in category_ids:
        print(f"\n--- Processing Category: {category_id} ---")

        # --- Filter by Category ID ---
        df['category_ids'] = df['category_ids'].astype(str)
        category_df = df[df['category_ids'].str.contains(fr'\b{re.escape(category_id)}\b', na=False)]

        if category_df.empty:
            print(f"No data found for category_id '{category_id}'.")
            continue
        
        # --- Filter out already sampled flaws ---
        available_category_df = category_df[~category_df['unique_flaw_id'].isin(sampled_flaw_ids)]

        # --- Separate into Mentioned and Not-Mentioned Groups ---
        available_mentioned = available_category_df[available_category_df['is_flaw_mentioned'] == True]
        available_not_mentioned = available_category_df[available_category_df['is_flaw_mentioned'] == False]
        
        print(f"Total entries for category '{category_id}': {len(category_df)}")
        print(f"  - Mentioned (True): Available for sampling={len(available_mentioned)}")
        print(f"  - Not Mentioned (False): Available for sampling={len(available_not_mentioned)}")

        # --- Sample from Each Available Group ---
        mentioned_sample = available_mentioned.sample(
            n=min(n_samples_per_group, len(available_mentioned)),
            random_state=seed
        )
        not_mentioned_sample = available_not_mentioned.sample(
            n=min(n_samples_per_group, len(available_not_mentioned)),
            random_state=seed
        )

        print(f"Sampling {len(mentioned_sample)} 'True' and {len(not_mentioned_sample)} 'False' entries.")
        
        # --- Update the set of sampled flaws ---
        sampled_flaw_ids.update(mentioned_sample['unique_flaw_id'])
        sampled_flaw_ids.update(not_mentioned_sample['unique_flaw_id'])

        # --- Combine and add to our list of samples ---
        if not mentioned_sample.empty or not not_mentioned_sample.empty:
            category_sample_df = pd.concat([mentioned_sample, not_mentioned_sample])
            all_samples_list.append(category_sample_df)

    if not all_samples_list:
        print("\nNo samples were collected across all categories.")
        return pd.DataFrame()

    # --- Combine all samples and clean up ---
    final_df = pd.concat(all_samples_list).reset_index(drop=True)
    # Drop the helper column
    final_df.drop(columns=['unique_flaw_id'], inplace=True)
    
    print(f"\nTotal unique samples collected: {len(final_df)}")
    
    # --- New filtering step based on options ---
    if not require_llm_review:
        final_df.drop(columns=['llm_review'], inplace=True)
        print(f"Filtered for LLM reviews.")
    if not (require_metareview and require_llm_review):
        final_df.drop(columns=['is_flaw_mentioned'], inplace=True)
        final_df.drop(columns=['mention_reasoning'], inplace=True)
        print(f"Filtered for Metareviews.")
    
    return final_df

def copy_paper_data(sampled_df, base_data_dir, venue_folder_name, venue, destination_base_dir):
    """
    Copies the 'flawed_papers' and 'original_papers' directories for the sampled flaws.
    """
    print("\n--- Starting to copy paper data ---")

    # Define source and destination paths
    source_flawed_papers_dir = os.path.join(base_data_dir, 'flawed_papers', venue_folder_name)
    original_venue_folder = f'{venue}_latest'
    source_original_papers_dir = os.path.join(base_data_dir, 'original_papers', original_venue_folder)

    dest_flawed_papers_dir = os.path.join(destination_base_dir, 'flawed_papers')
    dest_original_papers_dir = os.path.join(destination_base_dir, 'original_papers')
    os.makedirs(dest_flawed_papers_dir, exist_ok=True)
    os.makedirs(dest_original_papers_dir, exist_ok=True)
    print(f"Ensured destination directories exist: {dest_flawed_papers_dir} and {dest_original_papers_dir}")

    copied_flawed_count = 0
    copied_original_count = 0
    unique_ids = sampled_df['openreview_id'].unique()

    for openreview_id in unique_ids:
        paper_folder_name = None
        status_found = None

        # Find the full paper folder name and its status (accepted/rejected)
        for status in ['accepted', 'rejected']:
            status_path = os.path.join(source_flawed_papers_dir, status)
            if not os.path.isdir(status_path):
                continue
            
            for folder in os.listdir(status_path):
                if folder.startswith(openreview_id) and os.path.isdir(os.path.join(status_path, folder)):
                    paper_folder_name = folder
                    status_found = status
                    break
            if paper_folder_name:
                break
        
        if not paper_folder_name:
            print(f"Warning: Could not find a source directory for openreview_id '{openreview_id}' in '{source_flawed_papers_dir}'")
            continue

        # --- 1. Copy 'flawed_papers' subdirectory ---
        source_flaws_subfolder = os.path.join(source_flawed_papers_dir, status_found, paper_folder_name, 'flawed_papers')
        dest_flaws_path = os.path.join(dest_flawed_papers_dir, paper_folder_name.split('_')[0])
        
        if os.path.isdir(source_flaws_subfolder):
            if os.path.exists(dest_flaws_path):
                print(f"Skipping 'flawed_papers' copy, destination already exists: {dest_flaws_path}")
            else:
                try:
                    shutil.copytree(source_flaws_subfolder, dest_flaws_path)
                    print(f"Copied '{source_flaws_subfolder}' to '{dest_flaws_path}'")
                    copied_flawed_count += 1
                except Exception as e:
                    print(f"Error copying {source_flaws_subfolder}: {e}")
        else:
             print(f"Warning: 'flawed_papers' subfolder not found at: {source_flaws_subfolder}")

        # --- 2. Copy 'original_papers' directory ---
        source_original_folder = os.path.join(source_original_papers_dir, status_found, paper_folder_name)
        dest_original_path = os.path.join(dest_original_papers_dir, paper_folder_name.split('_')[0])

        if os.path.isdir(source_original_folder):
             if os.path.exists(dest_original_path):
                print(f"Skipping 'original_papers' copy, destination already exists: {dest_original_path}")
             else:
                try:
                    shutil.copytree(source_original_folder, dest_original_path)
                    print(f"Copied '{source_original_folder}' to '{dest_original_path}'")
                    copied_original_count += 1
                except Exception as e:
                    print(f"Error copying {source_original_folder}: {e}")
        else:
            print(f"Warning: 'original_papers' folder not found at: {source_original_folder}")

    print(f"\nFinished copying.")
    print(f" - Copied 'flawed_papers' directories for {copied_flawed_count} papers.")
    print(f" - Copied 'original_papers' directories for {copied_original_count} papers.")


if __name__ == '__main__':
    # --- Configuration ---
    VENUE = 'NeurIPS2024'
    VENUE_FOLDER_NAME = f'{VENUE}_latest_flawed_papers_v1'
    BASE_DATA_DIRECTORY = '../../data'
    CATEGORIZED_DATA_DIRECTORY = '../../data/categories_of_consensus_flaws'
    SAMPLES_PER_GROUP = 1
    ALL_CATEGORY_IDS_TO_SAMPLE = [
        '1a', '1b', '1c', '1d', '2a', '2b', '2c', '3a', '3b',
        '4a', '4b', '5a', '5b'
    ]
    EXP_DATA_DIR = f'./exp_data_{SAMPLES_PER_GROUP}_per_group_{VENUE}'
    FULL_FLAWS_FILENAME = f'{VENUE}_all_flaws.csv'

    os.makedirs(EXP_DATA_DIR, exist_ok=True)
    create_aggregated_dataset(VENUE_FOLDER_NAME, BASE_DATA_DIRECTORY, CATEGORIZED_DATA_DIRECTORY, os.path.join(EXP_DATA_DIR, FULL_FLAWS_FILENAME))

    # --- SAMPLING EXECUTION ---
    
    
    OUTPUT_SAMPLED_CSV_FILENAME = f'sampled_flaws.csv'
    
    # --- New Options ---
    REQUIRE_LLM_REVIEW = False
    REQUIRE_METAREVIEW = False
    COPY_SAMPLED_PAPERS = True
    SEED = 42

    print(f"\n--- Starting Sampling Process for Categories: {ALL_CATEGORY_IDS_TO_SAMPLE} ---")
    
    sampled_df = sample_flaw_data(
        os.path.join(EXP_DATA_DIR, FULL_FLAWS_FILENAME),
        ALL_CATEGORY_IDS_TO_SAMPLE,
        SAMPLES_PER_GROUP,
        require_llm_review=REQUIRE_LLM_REVIEW,
        require_metareview=REQUIRE_METAREVIEW,
        seed=SEED
    )

    if sampled_df is not None and not sampled_df.empty:
        output_path = os.path.join(EXP_DATA_DIR, OUTPUT_SAMPLED_CSV_FILENAME)
        sampled_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved {len(sampled_df)} unique sampled rows to '{output_path}'.")

        if COPY_SAMPLED_PAPERS:
            copy_paper_data(
                sampled_df=sampled_df,
                base_data_dir=BASE_DATA_DIRECTORY,
                venue_folder_name=VENUE_FOLDER_NAME,
                venue=VENUE,
                destination_base_dir=EXP_DATA_DIR
            )
    else:
        print("\nNo final sample was generated.")
        
    os.remove(os.path.join(EXP_DATA_DIR, FULL_FLAWS_FILENAME))

