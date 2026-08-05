from dataclasses import dataclass
from typing import Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class RAGHyperParams(HyperParams):
    """Hyperparameters for RAG (Retrieval-Augmented Generation) knowledge editing."""
    
    # Model configuration
    device: int
    alg_name: str
    model_name: str
    
    # RAG-specific parameters
    retriever_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 5  # Number of retrieved examples
    use_retrieved_examples: bool = True
    max_retrieved_tokens: int = 512
    
    # Retrieval configuration
    retrieval_batch_size: int = 32
    embedding_cache_dir: str = "./cache/rag/embeddings"
    
    # Generation configuration
    max_gen_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Results and logging
    results_dir: str = "./results"
    model_parallel: bool = False
    max_length: int = 40  # Standard tokenization max length
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        """Load hyperparameters from a YAML file."""
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'
        
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        
        assert (config and config['alg_name'] == 'RAG') or print(
            f'RAGHyperParams cannot load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]}'
        )
        return cls(**config)
