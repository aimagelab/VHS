import os
import random
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple
import torch

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible experiments.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy random number generator
    - PyTorch random number generators (CPU and GPU)
    - PyTorch's CUDNN backend for deterministic operations
    - Transformers library (if available)
    - Environment variables for hash randomization
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For transformers library
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass
    
    print(f"Random seed set to {seed} for reproducible experiments")


def subsample_by_unique_reason(data_list, sample_size=3):
    """
    Selects up to `sample_size` samples with unique reasons from the given list.
    
    :param data_list: List of dictionaries with 'path' and 'reason' keys
    :param sample_size: Number of unique samples to select (default is 3)
    :return: List of selected dictionaries
    """
    reason_map = {}
    for item in data_list:
        reason = item['reason']
        if reason not in reason_map:
            reason_map[reason] = item
    
    unique_samples = np.array(list(reason_map.values()))
    return list(np.random.choice(unique_samples, min(sample_size, len(unique_samples)), replace=False))


class Verifier(ABC):
    """
    Abstract base class for verifiers.
    
    Subclasses should implement the `verify` method.
    """
    
    @abstractmethod
    def verify(self, image, text):
        """
        Verify the given image and text.
        
        :param image: Image to verify
        :param text: Text to verify
        :return: Verification result
        """
        pass
    @abstractmethod
    def get_top_k(self, base_path, k, b, batch_size):
        """
        Get the top k results based on the verification.
        
        :param base_path: Base path for the results
        :param k: Number of top results to return
        :param b: Some parameter related to the model
        :param model: Model used for verification
        :return: Top k results
        """
        pass


