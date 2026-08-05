"""
RAG (Retrieval-Augmented Generation) Knowledge Editing Method

This implementation augments the model's knowledge by retrieving relevant
examples from a knowledge base and using them to inform the generation process.
The retrieval is based on semantic similarity to the edit request.
"""

import os
import json
import pickle
import torch
from typing import Any, Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

from .rag_hparams import RAGHyperParams


class RAGRetriever:
    """Handles retrieval of relevant examples for RAG-based editing."""
    
    def __init__(self, hparams: RAGHyperParams, device: int):
        self.hparams = hparams
        self.device = torch.device(f'cuda:{device}')
        self.retriever = SentenceTransformer(hparams.retriever_model_name).to(self.device)
        self.knowledge_base = []
        self.embeddings = None
        
    def build_knowledge_base(self, dataset) -> None:
        """Build knowledge base from dataset."""
        print(f"[RAG] Building knowledge base from {len(dataset)} examples...")
        self.knowledge_base = []
        
        for idx, example in enumerate(dataset):
            if idx % 1000 == 0:
                print(f"[RAG] Processing example {idx}/{len(dataset)}")
            
            # Format example for knowledge base
            fact_text = f"Prompt: {example['prompt']}\nAnswer: {example['target_new']}"
            self.knowledge_base.append({
                'text': fact_text,
                'prompt': example['prompt'],
                'target_new': example['target_new'],
                'target_old': example.get('ground_truth', ''),
                'idx': idx
            })
        
        print(f"[RAG] Knowledge base built with {len(self.knowledge_base)} entries")
        
    def compute_embeddings(self) -> None:
        """Compute and cache embeddings for knowledge base."""
        if len(self.knowledge_base) == 0:
            raise ValueError("Knowledge base is empty. Call build_knowledge_base first.")
        
        cache_dir = self.hparams.embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        safe_model_name = self.hparams.retriever_model_name.rsplit('/', 1)[-1]
        cache_file = os.path.join(cache_dir, f"rag_embeddings_{safe_model_name}.pkl")
        
        if os.path.exists(cache_file):
            print(f"[RAG] Loading embeddings from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                self.embeddings = torch.tensor(data['embeddings']).to(self.device)
            return
        
        print(f"[RAG] Computing embeddings for {len(self.knowledge_base)} examples...")
        texts = [item['text'] for item in self.knowledge_base]
        
        # Compute embeddings in batches
        embeddings_list = []
        batch_size = self.hparams.retrieval_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.retriever.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = torch.cat(embeddings_list, dim=0).to(self.device)
        self.embeddings = util.normalize_embeddings(self.embeddings)
        
        # Cache embeddings
        with open(cache_file, "wb") as f:
            pickle.dump({
                'embeddings': self.embeddings.cpu().numpy(),
                'knowledge_base': self.knowledge_base
            }, f)
        
        print(f"[RAG] Embeddings cached to: {cache_file}")
        
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve top-k relevant examples for a query."""
        if k is None:
            k = self.hparams.k
        
        if self.embeddings is None or len(self.knowledge_base) == 0:
            return []
        
        # Encode query
        query_embedding = self.retriever.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        ).to(self.device)
        query_embedding = util.normalize_embeddings(query_embedding.unsqueeze(0))
        
        # Retrieve similar examples
        hits = util.semantic_search(
            query_embedding,
            self.embeddings,
            score_function=util.dot_score,
            top_k=min(k, len(self.knowledge_base))
        )
        
        retrieved_examples = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            retrieved_examples.append({
                **self.knowledge_base[idx],
                'score': hit['score']
            })
        
        return retrieved_examples


def apply_rag_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: RAGHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Apply RAG knowledge editing to the model.
    
    Instead of modifying model weights directly, RAG augments generation with
    retrieved examples that are inserted into the prompt as in-context learning.
    
    Args:
        model: The language model to edit
        tok: The tokenizer
        request: Edit request containing prompt, target_new, etc.
        hparams: RAG hyperparameters
        train_ds: Training dataset for building knowledge base
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, editing_result)
    """
    if isinstance(request, list):
        request = request[0]
    
    device = torch.device(f'cuda:{hparams.device}')
    
    # Initialize retriever
    retriever = RAGRetriever(hparams, hparams.device)
    
    # Build knowledge base if provided
    if train_ds is not None:
        retriever.build_knowledge_base(train_ds)
        retriever.compute_embeddings()
    else:
        print("[RAG] Warning: No training dataset provided for knowledge base")
        return model, {}
    
    # Create query from the edit request
    new_fact = f"{request['prompt']} {request['target_new']}"
    
    if hparams.use_retrieved_examples:
        retrieved = retriever.retrieve(new_fact, k=hparams.k)
        
        # Format retrieved examples as in-context learning prompt
        in_context_prompt = ""
        for i, example in enumerate(retrieved, 1):
            in_context_prompt += f"Example {i}:\n{example['text']}\n\n"
        
        result = {
            'in_context_prompt': in_context_prompt,
            'retrieved_examples': retrieved,
            'num_retrieved': len(retrieved),
        }
    else:
        result = {
            'in_context_prompt': '',
            'retrieved_examples': [],
            'num_retrieved': 0,
        }
    
    return model, result


def apply_rag_to_multimodal_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: RAGHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """Apply RAG to multimodal models."""
    # For multimodal, use the same logic as standard RAG
    return apply_rag_to_model(
        model, tok, request, hparams,
        copy=copy,
        return_orig_weights=return_orig_weights,
        keep_original_weight=keep_original_weight,
        train_ds=train_ds,
        **kwargs
    )
