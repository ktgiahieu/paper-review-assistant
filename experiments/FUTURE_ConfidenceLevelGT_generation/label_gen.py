import pandas as pd
import re
import json
from io import StringIO
from collections import defaultdict

def extract_flaw_quotes(reasoning_text):
    """
    Extracts top-level sentences quoted in the reasoning text.
    Handles multiple patterns in order of precedence:
    1. Text enclosed in double-double quotes (""..."").
    2. Text enclosed in standard double quotes ("...").
    3. Unclosed quotes that are introduced by a colon and run to the end of a sentence.

    Args:
        reasoning_text (str): The text explaining why a flaw was mentioned.
    Returns:
        list: A list of quoted strings.
    """
    # Regex to find three types of quotes. The order is important to ensure
    # the most specific patterns are matched first.
    # 1. ""(.*?)""      : Non-greedy match for text in double-double quotes.
    # 2. "([^"]+)"     : Match for text in standard quotes (ensuring it's not empty and has no quotes inside).
    # 3. (?<=:\s)"(.*?)(?=\.|$) : Match for an unclosed quote after a colon, capturing until a period or end of string.
    regex = r'""(.*?)""|"([^"]+)"|(?<=:\s)"(.*?)(?=\.|$)'
    
    matches = re.findall(regex, reasoning_text, re.DOTALL)
    
    # re.findall with '|' returns tuples of capturing groups. We need to extract the non-empty match from each tuple.
    processed_quotes = []
    for match_tuple in matches:
        # Find the first non-empty string in the tuple
        found_quote = next((s for s in match_tuple if s), None)
        if found_quote:
            processed_quotes.append(found_quote.strip())

    # Filter for more substantial quotes (more than 3 words).
    filtered_matches = [
        match for match in processed_quotes 
        if len(match.split()) > 3 and match.strip()
    ]
    
    # Fallback to the original list if the filter was too aggressive.
    if not filtered_matches and processed_quotes:
        return [q for q in processed_quotes if q.strip()]
        
    return filtered_matches

def find_spans(quote_text, full_review_text):
    """
    Finds all occurrences of a quote in the full text, using both exact and fuzzy matching.
    
    Args:
        quote_text (str): The text snippet to find.
        full_review_text (str): The text to search within.
        
    Returns:
        list: A list of (start, end) character spans.
    """
    spans = []

    # 1. Try exact match first
    cleaned_quote = quote_text.strip()
    if cleaned_quote:
        for match in re.finditer(re.escape(cleaned_quote), full_review_text):
            spans.append((match.start(), match.end()))
    
    if spans:
        return spans # Return if exact match is found

    # 2. If no exact match, fall back to a fuzzy match to handle formatting differences
    words = [re.escape(w) for w in re.split(r'\s+', quote_text.strip()) if w]
    if not words:
        return []

    # Separator allows for whitespace, markdown characters, and different quote styles
    separator = r"[\s\*'\"“”‘’]+"
    pattern = separator.join(words)
    
    normalized_quote_text = re.sub(r"[\s\*'\"“”‘’]", "", quote_text).lower()

    for match in re.finditer(pattern, full_review_text, re.IGNORECASE):
        matched_text_from_review = match.group(0)
        normalized_matched_text = re.sub(r"[\s\*'\"“”‘’]", "", matched_text_from_review).lower()
        
        # Check if the core content matches to avoid overly broad matches
        if normalized_quote_text in normalized_matched_text and len(normalized_matched_text) < len(normalized_quote_text) * 1.8:
            spans.append((match.start(), match.end()))
            
    return spans


def create_ner_training_data(df):
    """
    Processes the raw DataFrame to create a labeled dataset for NER training.
    It groups all flaw spans for each unique review.
    
    Args:
        df (pd.DataFrame): The input DataFrame with review data.
        
    Returns:
        pd.DataFrame: A DataFrame with columns ['review_text', 'spans'], where 'spans'
                      is a list of (start_char, end_char) tuples.
    """
    reviews_with_spans = defaultdict(list)

    for _, row in df.iterrows():
        llm_review = row['llm_review']
        is_flaw_mentioned = row['is_flaw_mentioned']
        mention_reasoning = str(row['mention_reasoning'])

        if pd.isna(llm_review):
            continue

        try:
            review_json = json.loads(llm_review)
            full_review_text = " ".join(str(v) for v in review_json.values())
        except (json.JSONDecodeError, TypeError):
            full_review_text = str(llm_review)

        if is_flaw_mentioned:
            quotes = extract_flaw_quotes(mention_reasoning)
            for quote in quotes:
                # Handle quotes ending in '...' or '…'
                if '...' in quote[-10:] or '…' in quote[-10:]:
                    ellipsis_char = '…' if '…' in quote else '...'
                    prefix = quote.split(ellipsis_char)[0].strip()
                    if not prefix: continue

                    for prefix_match in re.finditer(re.escape(prefix), full_review_text):
                        start_index = prefix_match.start()
                        sentence_end_match = re.search(r'[\.?!]', full_review_text[start_index:])
                        end_index = start_index + sentence_end_match.end() if sentence_end_match else len(full_review_text)
                        
                        span = (start_index, end_index)
                        if span not in reviews_with_spans[full_review_text]:
                            reviews_with_spans[full_review_text].append(span)
                
                # Handle normal quotes with the new robust find_spans function
                else:
                    found_spans = find_spans(quote, full_review_text)
                    for span in found_spans:
                        if span not in reviews_with_spans[full_review_text]:
                            reviews_with_spans[full_review_text].append(span)

    output_data = [
        {'review_text': text, 'spans': spans}
        for text, spans in reviews_with_spans.items() if spans
    ]

    return pd.DataFrame(output_data)


if __name__ == '__main__':
    # Use StringIO to simulate reading the CSV data provided in the prompt
    df = pd.read_csv('../data/neurips_2024_aggregated_flaws.csv')
    
    # Create the labeled training data
    labeled_df = create_ner_training_data(df)
    
    # Save to a file
    labeled_df.to_csv('labeled_flaws_ner.csv', index=False, escapechar='\\')
    
    print("Generated labeled data and saved to 'labeled_flaws.csv'")
    print(f"Total samples: {len(labeled_df)}")
    print("Sample data:")
    print(labeled_df.head())
