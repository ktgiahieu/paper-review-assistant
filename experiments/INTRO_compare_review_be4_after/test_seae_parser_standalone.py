#!/usr/bin/env python3
"""
Standalone test script for SEA-E parser (no external dependencies)
"""

import re
import json

def parse_seae_format(content: str) -> dict:
    """
    Parses SEA-E markdown format into structured dictionary.
    """
    result = {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "soundness": "",
        "presentation": "",
        "contribution": "",
        "rating": "",
        "paper_decision": "",
    }
    
    # Split content by section headers (handles multi-word headers like "Paper Decision")
    sections = re.split(r'\*\*([^*:]+):\*\*', content)
    
    # Process sections (odd indices are headers, even indices are content)
    for i in range(1, len(sections), 2):
        if i+1 < len(sections):
            section_name = sections[i].lower().strip()
            section_content = sections[i+1].strip()
            
            if section_name == "summary":
                result["summary"] = section_content
            
            elif section_name == "strengths":
                strengths = re.findall(r'^\s*-\s*(.+)$', section_content, re.MULTILINE)
                result["strengths"] = [s.strip() for s in strengths if s.strip()]
            
            elif section_name == "weaknesses":
                weaknesses = re.findall(r'^\s*-\s*(.+)$', section_content, re.MULTILINE)
                result["weaknesses"] = [w.strip() for w in weaknesses if w.strip()]
            
            elif section_name == "questions":
                questions = re.findall(r'^\s*-\s*(.+)$', section_content, re.MULTILINE)
                result["questions"] = [q.strip() for q in questions if q.strip()]
            
            elif section_name == "soundness":
                result["soundness"] = section_content
            
            elif section_name == "presentation":
                result["presentation"] = section_content
            
            elif section_name == "contribution":
                result["contribution"] = section_content
            
            elif section_name == "rating":
                result["rating"] = section_content
            
            elif section_name in ["paper decision", "paperdecision", "decision"]:
                decision_match = re.search(r'-\s*Decision:\s*(\w+)', section_content, re.IGNORECASE)
                reasons_match = re.search(r'-\s*Reasons?:\s*(.+)', section_content, re.IGNORECASE | re.DOTALL)
                
                decision = decision_match.group(1) if decision_match else "Unknown"
                reasons = reasons_match.group(1).strip() if reasons_match else section_content
                
                result["paper_decision"] = f"Decision: {decision}\nReasons: {reasons}"
    
    return result

# Example SEA-E output
EXAMPLE_OUTPUT = """**Summary:**

The paper investigates the efficient computation of the attention matrix in linear time, a critical task in large language models (LLMs). It demonstrates that when the entries of matrices in the attention operation are bounded, it is possible to compute the attention matrix in near-linear time.



**Strengths:**

- The paper introduces a novel approach that transforms entry-wise exponential attention operators into low-rank low-degree polynomial approximations.

- It verifies the phenomenon that small entries in attention computation can lead to faster attention computation.

- The theoretical analysis is solid and the algorithm is clear.



**Weaknesses:**

- The paper's scope is narrow and specific, focusing solely on attention computation.

- The algorithm design is somewhat standard and unsurprising.

- The paper's assumptions about the dimensions limit the generalizability of the findings.



**Questions:**

- Have the authors implemented the presented algorithm?

- Could the authors clarify the assumptions in Lemma 12?

- How does the algorithm handle practical aspects?



**Soundness:**

3 good



**Presentation:**

3 good



**Contribution:**

3 good



**Rating:**

6 weak accept



**Paper Decision:**

- Decision: Accept

- Reasons: The paper presents a novel approach to computing the attention matrix in nearly linear time, which is significant for large language models. The theoretical contributions are robust, and the methodology provides a novel perspective on the problem.</s>
"""

def main():
    print("=" * 80)
    print("Testing SEA-E Parser (Standalone)")
    print("=" * 80)
    
    # Parse the example
    result = parse_seae_format(EXAMPLE_OUTPUT)
    
    # Print results
    print("\n" + "=" * 80)
    print("PARSED RESULTS")
    print("=" * 80)
    
    print(f"\n📝 SUMMARY:")
    print(f"   {result['summary'][:150]}...")
    
    print(f"\n✅ STRENGTHS ({len(result['strengths'])} items):")
    for i, strength in enumerate(result['strengths'], 1):
        print(f"   {i}. {strength}")
    
    print(f"\n⚠️  WEAKNESSES ({len(result['weaknesses'])} items):")
    for i, weakness in enumerate(result['weaknesses'], 1):
        print(f"   {i}. {weakness}")
    
    print(f"\n❓ QUESTIONS ({len(result['questions'])} items):")
    for i, question in enumerate(result['questions'], 1):
        print(f"   {i}. {question}")
    
    print(f"\n📊 RATINGS:")
    print(f"   Soundness:     {result['soundness']}")
    print(f"   Presentation:  {result['presentation']}")
    print(f"   Contribution:  {result['contribution']}")
    print(f"   Overall:       {result['rating']}")
    
    print(f"\n🎯 PAPER DECISION:")
    for line in result['paper_decision'].split('\n')[:5]:
        print(f"   {line}")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks = {
        "Summary exists": bool(result['summary']),
        "Has 3 strengths": len(result['strengths']) == 3,
        "Has 3 weaknesses": len(result['weaknesses']) == 3,
        "Has 3 questions": len(result['questions']) == 3,
        "Soundness = '3 good'": result['soundness'] == "3 good",
        "Presentation = '3 good'": result['presentation'] == "3 good",
        "Contribution = '3 good'": result['contribution'] == "3 good",
        "Rating = '6 weak accept'": result['rating'] == "6 weak accept",
        "Decision contains 'Accept'": "Accept" in result['paper_decision'],
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {check}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("The SEA-E parser is working correctly!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review parser logic")
    print("=" * 80)
    
    # Print JSON
    print("\nFull JSON output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

