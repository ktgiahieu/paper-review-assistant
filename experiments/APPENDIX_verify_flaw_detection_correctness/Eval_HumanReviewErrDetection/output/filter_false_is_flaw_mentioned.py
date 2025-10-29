def filter_false_is_flaw_mentioned(entities: list[dict]) -> list[dict]:
    """
    Filters a list of entities, returning only those where 'is_flaw_mentioned' is True.

    Args:
        entities: A list of dictionaries, where each dictionary represents an entity
                  and contains a key 'is_flaw_mentioned' with a boolean value.

    Returns:
        A new list containing only the entities where 'is_flaw_mentioned' is False.
    """
    return [entity for entity in entities if not entity['evaluation']['is_flaw_mentioned']]

import json
with open('./sonnet_reviews_evaluated_NeurIPS2024.json', 'r') as f:
    entities = json.load(f)
    
filtered_false_entities = filter_false_is_flaw_mentioned(entities=entities)

with open('./false_scenario_review_later.json', 'w') as f:
    json.dump(filtered_false_entities, f, ensure_ascii=False, indent=2)