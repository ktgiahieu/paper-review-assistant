#!/usr/bin/env python3
"""
Utility script to get v1 versions of papers that match papers in the latest folder.

This script:
1. Reads paper IDs from the latest folder (e.g., sampled_data/ICLR2024/latest/)
2. Finds matching v1 versions in the v1 folder (e.g., data/ICLR2024_pairs/v1/)
3. Copies the matching v1 papers to create pairs alongside the latest versions

Usage:
    python3 get_matching_v1_pairs.py \
        --latest_dir ../../sampled_data/ICLR2024/latest \
        --v1_source_dir ../../data/ICLR2024_pairs/v1 \
        --output_dir ../../sampled_data/ICLR2024/v1
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def extract_paper_id(folder_name: str) -> str:
    """
    Extract paper ID from folder name.
    
    Examples:
        "ViNe1fjGME_2305_10738" -> "ViNe1fjGME"
        "2Rwq6c3tvr_2308_08493v1" -> "2Rwq6c3tvr"
        "0akLDTFR9x_2310_20141v1" -> "0akLDTFR9x"
    """
    # Remove v1 suffix if present
    if folder_name.endswith('v1'):
        folder_name = folder_name[:-2]
    
    # Extract paper ID (everything before the first underscore followed by numbers)
    parts = folder_name.split('_')
    if parts:
        return parts[0]
    return folder_name


def find_matching_v1_folder(paper_id: str, v1_source_dir: Path) -> Path:
    """
    Find the v1 folder that matches the given paper ID.
    
    Args:
        paper_id: Paper ID to match
        v1_source_dir: Directory containing v1 folders
        
    Returns:
        Path to matching v1 folder, or None if not found
    """
    if not v1_source_dir.exists():
        return None
    
    # Look for folders that start with the paper_id and end with 'v1'
    for folder in v1_source_dir.iterdir():
        if not folder.is_dir():
            continue
        
        folder_paper_id = extract_paper_id(folder.name)
        if folder_paper_id == paper_id:
            return folder
    
    return None


def get_matching_v1_pairs(
    latest_dir: Path,
    v1_source_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> dict:
    """
    Get v1 versions that match papers in the latest folder.
    
    Args:
        latest_dir: Directory containing latest versions
        v1_source_dir: Directory containing v1 source versions
        output_dir: Output directory for v1 versions (will create pairs)
        verbose: Enable verbose output
        
    Returns:
        Dictionary with matching statistics
    """
    latest_dir = Path(latest_dir)
    v1_source_dir = Path(v1_source_dir)
    output_dir = Path(output_dir)
    
    if not latest_dir.exists():
        raise ValueError(f"Latest directory does not exist: {latest_dir}")
    
    if not v1_source_dir.exists():
        raise ValueError(f"V1 source directory does not exist: {v1_source_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all latest paper folders
    latest_folders = [d for d in latest_dir.iterdir() if d.is_dir()]
    print(f"Found {len(latest_folders)} papers in latest/")
    
    # Extract paper IDs from latest folders
    latest_paper_ids = {}
    for folder in latest_folders:
        paper_id = extract_paper_id(folder.name)
        latest_paper_ids[paper_id] = folder.name
    
    print(f"Extracted {len(latest_paper_ids)} unique paper IDs")
    
    # Find matching v1 versions
    matched = []
    not_found = []
    
    print("\nFinding matching v1 versions...")
    for paper_id, latest_folder_name in tqdm(latest_paper_ids.items(), desc="Matching papers"):
        v1_folder = find_matching_v1_folder(paper_id, v1_source_dir)
        
        if v1_folder:
            matched.append({
                'paper_id': paper_id,
                'latest_folder': latest_folder_name,
                'v1_folder': v1_folder.name,
                'v1_path': v1_folder
            })
        else:
            not_found.append(paper_id)
            if verbose:
                print(f"  ⚠️  No v1 version found for {paper_id}")
    
    print(f"\n✅ Found {len(matched)} matching pairs")
    if not_found:
        print(f"⚠️  {len(not_found)} papers have no v1 version: {', '.join(not_found[:10])}")
        if len(not_found) > 10:
            print(f"    ... and {len(not_found) - 10} more")
    
    # Copy matching v1 versions to output directory
    if matched:
        print(f"\nCopying {len(matched)} v1 versions to {output_dir}/...")
        copied = 0
        skipped = 0
        
        for match in tqdm(matched, desc="Copying v1 versions"):
            source = match['v1_path']
            dest = output_dir / match['v1_folder']
            
            if dest.exists():
                if verbose:
                    print(f"  ⏭️  Skipping {match['paper_id']} (already exists)")
                skipped += 1
                continue
            
            try:
                shutil.copytree(source, dest)
                copied += 1
                if verbose:
                    print(f"  ✅ Copied {match['paper_id']}: {source.name} -> {dest.name}")
            except Exception as e:
                print(f"  ❌ Error copying {match['paper_id']}: {e}")
        
        print(f"\n✅ Copied {copied} v1 versions")
        if skipped > 0:
            print(f"⏭️  Skipped {skipped} (already exist)")
    
    return {
        'total_latest': len(latest_paper_ids),
        'matched': len(matched),
        'not_found': len(not_found),
        'not_found_ids': not_found,
        'copied': copied if matched else 0,
        'skipped': skipped if matched else 0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Get v1 versions of papers that match papers in the latest folder"
    )
    parser.add_argument(
        "--latest_dir",
        type=str,
        required=True,
        help="Directory containing latest paper versions (e.g., sampled_data/ICLR2024/latest)"
    )
    parser.add_argument(
        "--v1_source_dir",
        type=str,
        required=True,
        help="Directory containing v1 source versions (e.g., data/ICLR2024_pairs/v1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for v1 versions (e.g., sampled_data/ICLR2024/v1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Get Matching V1 Pairs")
    print("="*80)
    print(f"Latest directory: {args.latest_dir}")
    print(f"V1 source directory: {args.v1_source_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        stats = get_matching_v1_pairs(
            latest_dir=args.latest_dir,
            v1_source_dir=args.v1_source_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"Total papers in latest/: {stats['total_latest']}")
        print(f"Matching v1 versions found: {stats['matched']}")
        print(f"Papers without v1 version: {stats['not_found']}")
        print(f"V1 versions copied: {stats['copied']}")
        print(f"V1 versions skipped (already exist): {stats['skipped']}")
        print("="*80)
        
        if stats['not_found'] > 0:
            print(f"\n⚠️  Papers without v1 versions:")
            for paper_id in stats['not_found_ids']:
                print(f"  - {paper_id}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

