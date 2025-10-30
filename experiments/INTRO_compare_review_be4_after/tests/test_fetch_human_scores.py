#!/usr/bin/env python3
"""
Test fetch_human_scores.py with OpenReview API

Quick test to verify the API is working correctly.

Usage:
    python test_fetch_human_scores.py
"""

import requests
import json

# OpenReview API base URL
OPENREVIEW_API_BASE = "https://api2.openreview.net"

# Test with a known ICLR 2024 paper (first paper from your CSV)
TEST_PAPER_ID = "ViNe1fjGME"

def test_api_v2_approach():
    """Test the API v2 approach from OpenReview docs."""
    
    print("="*80)
    print("Testing OpenReview API v2 - Correct Approach")
    print("="*80)
    print(f"\nTest paper ID: {TEST_PAPER_ID}")
    print("Paper: Deep Temporal Graph Clustering")
    
    # Get submission with replies
    url = f"{OPENREVIEW_API_BASE}/notes"
    params = {
        "id": TEST_PAPER_ID,
        "details": "replies"
    }
    
    print(f"\nQuerying: GET {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'notes' not in data or len(data['notes']) == 0:
            print("\n❌ No submission found!")
            return False
        
        submission = data['notes'][0]
        print(f"\n✅ Found submission:")
        print(f"  Title: {submission.get('content', {}).get('title', {}).get('value', 'N/A')}")
        print(f"  ID: {submission.get('id')}")
        
        # Check replies
        replies = submission.get('details', {}).get('replies', [])
        print(f"\n✅ Found {len(replies)} replies")
        
        # Filter for official reviews
        review_notes = []
        for reply in replies:
            invitations = reply.get('invitations', [])
            if any(inv.endswith('Official_Review') for inv in invitations):
                review_notes.append(reply)
        
        print(f"✅ Found {len(review_notes)} official reviews")
        
        if len(review_notes) == 0:
            print("\n⚠️  No official reviews found!")
            print("This might mean:")
            print("  - Paper doesn't have reviews yet")
            print("  - Reviews are not public")
            print("  - Different invitation pattern")
            return False
        
        # Extract scores from first review
        print(f"\n{'='*80}")
        print("Examining first review:")
        print('='*80)
        
        first_review = review_notes[0]
        print(f"\nInvitations: {first_review.get('invitations', [])}")
        
        content = first_review.get('content', {})
        print(f"\nContent keys: {list(content.keys())}")
        
        # Try to extract scores
        print(f"\n{'='*80}")
        print("Attempting to extract scores:")
        print('='*80)
        
        score_fields = {
            'rating': ['rating'],
            'soundness': ['soundness', 'technical_soundness', 'correctness'],
            'presentation': ['presentation', 'clarity'],
            'contribution': ['contribution', 'novelty'],
            'confidence': ['confidence']
        }
        
        found_scores = {}
        for score_name, field_names in score_fields.items():
            for field_name in field_names:
                if field_name in content:
                    value = content[field_name]
                    if isinstance(value, dict):
                        value = value.get('value', '')
                    
                    print(f"\n{score_name.upper()} (field: {field_name}):")
                    print(f"  Raw value: {str(value)[:150]}")
                    
                    # Try to parse score
                    if isinstance(value, str):
                        try:
                            score = float(value.split(':')[0].strip())
                            found_scores[score_name] = score
                            print(f"  ✅ Parsed score: {score}")
                        except:
                            print(f"  ⚠️  Could not parse as number")
                    elif isinstance(value, (int, float)):
                        found_scores[score_name] = float(value)
                        print(f"  ✅ Direct score: {value}")
                    
                    break
        
        print(f"\n{'='*80}")
        print("Summary:")
        print('='*80)
        print(f"\n✅ Successfully extracted {len(found_scores)} scores:")
        for score_name, score_value in found_scores.items():
            print(f"  {score_name}: {score_value}")
        
        if len(found_scores) >= 3:
            print(f"\n✅ SUCCESS! The API is working correctly.")
            print(f"   fetch_human_scores.py should work now.")
            return True
        else:
            print(f"\n⚠️  Only found {len(found_scores)} scores.")
            print(f"   This might be a score format issue.")
            return False
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("OpenReview API Test for fetch_human_scores.py")
    print("="*80)
    print("\nThis test verifies that:")
    print("  1. OpenReview API v2 is accessible")
    print("  2. Submissions can be retrieved with replies")
    print("  3. Official reviews can be identified")
    print("  4. Scores can be extracted from reviews")
    print()
    
    success = test_api_v2_approach()
    
    print("\n" + "="*80)
    if success:
        print("✅ TEST PASSED")
        print("="*80)
        print("\nYou can now run:")
        print("  python fetch_human_scores.py --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv --limit 5 --debug")
    else:
        print("⚠️  TEST HAD ISSUES")
        print("="*80)
        print("\nPlease check:")
        print("  1. Internet connection")
        print("  2. OpenReview API is accessible")
        print("  3. Paper IDs in CSV are correct")
        print("\nTry running with debug mode:")
        print("  python fetch_human_scores.py --csv_file ./data/ICLR2024_pairs/filtered_pairs.csv --limit 1 --debug")

if __name__ == "__main__":
    main()

