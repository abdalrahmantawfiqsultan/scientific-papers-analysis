from rapidfuzz import process, fuzz

def test_extract_one_finds_best_match():
    existing_entities = {
        "ent_1": "Neural Networks",
        "ent_2": "Deep Neural Networks for Vision",
        "ent_3": "Deep Neural Network"
    }
    
    extracted_clean = "deep neural network"
    
    from rapidfuzz import utils
    match = process.extractOne(extracted_clean, list(existing_entities.values()), scorer=fuzz.ratio, processor=utils.default_process)
    
    assert match is not None
    assert match[1] >= 85
    
    # It should pick "Deep Neural Network" over "Deep Neural Networks for Vision" 
    # even if "Deep Neural Networks for Vision" comes first alphabetically or insertion-wise
    assert match[0] == "Deep Neural Network"
    
    canonical_id = None
    for k, v in existing_entities.items():
        if v == match[0]:
            canonical_id = k
            break
            
    assert canonical_id == "ent_3"
