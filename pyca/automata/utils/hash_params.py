import os
import pickle

import torch
import hashlib
from typing import Dict


def params_to_words(state_dict: Dict[str, torch.Tensor], num_words: int = 2) -> str:
    """
    Convert a state dictionary into a deterministic sequence of words.
    Acts like a hash function - similar parameters will generate the same words.
    
    Args:
        state_dict: Model state dictionary loaded from torch.load()
        num_words: Number of words to generate
    
    Returns:
        String containing space-separated words
    """
    adj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adjectives.pk')
    noun_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nouns.pk')
    # Load adjectives and nouns
    with open(adj_path, 'rb') as handle:
        adjectives = pickle.load(handle)
    with open(noun_path, 'rb') as handle:
        nouns = pickle.load(handle)

    # Convert state dict to bytes for hashing
    param_bytes = b''
    for key in sorted(state_dict.keys()):  # Sort keys for deterministic ordering
        if not(isinstance(state_dict[key],torch.Tensor)):
            tohash = torch.tensor(state_dict[key])
        else:
            tohash = state_dict[key]
        # use torch.tensor as a hack, to convert the 'int' of k_size to bytes
        param_bytes += tohash.cpu().numpy().tobytes() 
    
    # Create deterministic seed from parameters
    hash_value = int(hashlib.sha256(param_bytes).hexdigest(), 16)
    
    # Generate words using the hash value
    words = []
    for i in range(num_words):
        seed = (hash_value + i * 12345) & 0xFFFFFFFF
        word_list = adjectives if i % 2 == 0 else nouns
        word_idx = seed % len(word_list)
        words.append(word_list[word_idx])
    
    return "_".join(words)
