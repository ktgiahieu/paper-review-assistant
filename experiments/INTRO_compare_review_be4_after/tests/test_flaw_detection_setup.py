#!/usr/bin/env python3
"""
Test Flaw Detection Setup

Quick test script to verify that your environment is correctly set up
for flaw detection evaluation.

Usage:
    python test_flaw_detection_setup.py \
      --evaluator_endpoint "http://localhost:8000" \
      --evaluator_model "Qwen3-30B-A3B-Instruct-2507-FP8"
"""

import argparse
import requests
import json

def test_evaluator_api(endpoint: str, model: str) -> bool:
    """Test if evaluator API is accessible and working."""
    print(f"Testing evaluator API at {endpoint}...")
    
    # Simple test prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Respond with: {'test': 'success'}"}
    ]
    
    request_data = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        print(f"✅ API Response: {content[:100]}...")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Cannot connect to {endpoint}")
        print(f"   Make sure vLLM server is running.")
        return False
    
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: API did not respond within 30 seconds")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_flaw_detection_logic(endpoint: str, model: str) -> bool:
    """Test the actual flaw detection logic."""
    print(f"\nTesting flaw detection logic...")
    
    # Test case
    ground_truth_flaw = "The paper presents results from only a single experimental run, which is insufficient for establishing statistical significance."
    
    weaknesses_positive = """
    - The experimental methodology lacks rigor as only one run was performed
    - No error bars or confidence intervals are provided
    - Results may not be reproducible
    """
    
    weaknesses_negative = """
    - The related work section is incomplete
    - Some figures are hard to read
    - The conclusion could be more detailed
    """
    
    system_prompt = """You are an expert evaluator. Determine if a ground truth flaw is mentioned in a review's weaknesses section.

Respond in JSON format:
{
  "detected": true/false,
  "reasoning": "Brief explanation"
}"""
    
    # Test 1: Should detect (positive case)
    print("\n  Test 1: Positive case (should detect)")
    user_prompt = f"""GROUND TRUTH FLAW:
{ground_truth_flaw}

WEAKNESSES SECTION:
{weaknesses_positive}

Is this flaw detected?"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    request_data = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=request_data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Parse JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        result_json = json.loads(content)
        detected = result_json.get('detected', False)
        reasoning = result_json.get('reasoning', '')
        
        if detected:
            print(f"  ✅ Correctly detected flaw")
            print(f"     Reasoning: {reasoning[:80]}...")
        else:
            print(f"  ⚠️  Failed to detect flaw (false negative)")
            print(f"     Reasoning: {reasoning[:80]}...")
        
    except Exception as e:
        print(f"  ❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Should not detect (negative case)
    print("\n  Test 2: Negative case (should not detect)")
    user_prompt = f"""GROUND TRUTH FLAW:
{ground_truth_flaw}

WEAKNESSES SECTION:
{weaknesses_negative}

Is this flaw detected?"""
    
    messages[1] = {"role": "user", "content": user_prompt}
    request_data["messages"] = messages
    
    try:
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=request_data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Parse JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        result_json = json.loads(content)
        detected = result_json.get('detected', False)
        reasoning = result_json.get('reasoning', '')
        
        if not detected:
            print(f"  ✅ Correctly did not detect flaw")
            print(f"     Reasoning: {reasoning[:80]}...")
        else:
            print(f"  ⚠️  Incorrectly detected flaw (false positive)")
            print(f"     Reasoning: {reasoning[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test 2 failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Test flaw detection setup"
    )
    parser.add_argument(
        "--evaluator_endpoint",
        type=str,
        default="http://localhost:8000",
        help="vLLM endpoint for evaluator model"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Evaluator model name"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Flaw Detection Setup Test")
    print("="*80)
    print(f"\nEvaluator endpoint: {args.evaluator_endpoint}")
    print(f"Evaluator model: {args.evaluator_model}")
    print()
    
    # Test 1: API accessibility
    api_ok = test_evaluator_api(args.evaluator_endpoint, args.evaluator_model)
    
    if not api_ok:
        print("\n" + "="*80)
        print("❌ Setup test FAILED")
        print("="*80)
        print("\nPlease fix the API connection issue before proceeding.")
        return
    
    # Test 2: Flaw detection logic
    logic_ok = test_flaw_detection_logic(args.evaluator_endpoint, args.evaluator_model)
    
    print("\n" + "="*80)
    if api_ok and logic_ok:
        print("✅ Setup test PASSED")
        print("="*80)
        print("\nYour environment is ready for flaw detection evaluation!")
        print("\nNext steps:")
        print("  1. Run: python evaluate_flaw_detection.py --csv_file ... --reviews_dir ...")
        print("  2. Run: python analyze_flaw_detection.py --results_file ...")
    else:
        print("⚠️  Setup test completed with warnings")
        print("="*80)
        print("\nThe API is accessible, but the flaw detection logic may need review.")
        print("You can proceed, but manually verify some results.")

if __name__ == "__main__":
    main()

