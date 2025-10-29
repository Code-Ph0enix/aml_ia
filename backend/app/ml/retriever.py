"""
Custom Retriever with E5-Base-V2 and FAISS
Trained with InfoNCE + Triplet Loss for banking domain

This is adapted from your RAG.py with:
- CustomSentenceTransformer (e5-base-v2)
- Mean pooling + L2 normalization
- FAISS vector search
- Module-level caching (load once on startup)
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel

from app.config import settings


# ============================================================================
# CUSTOM SENTENCE TRANSFORMER (From RAG.py)
# ============================================================================

class CustomSentenceTransformer(nn.Module):
    """
    Custom SentenceTransformer matching your training code.
    Uses e5-base-v2 with mean pooling and L2 normalization.
    
    Training Details:
    - Base model: intfloat/e5-base-v2
    - Loss: InfoNCE + Triplet Loss
    - Pooling: Mean pooling on last hidden state
    - Normalization: L2 normalization
    """
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        super().__init__()
        # Load pre-trained e5-base-v2 encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.encoder.config
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT encoder.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padding
        
        Returns:
            torch.Tensor: L2-normalized embeddings (shape: [batch_size, 768])
        """
        # Get BERT outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling - same as training
        # Take hidden states from last layer
        token_embeddings = outputs.last_hidden_state
        
        # Expand attention mask to match token embeddings shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings (weighted by attention mask) and divide by sum of mask
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        
        # L2 normalize embeddings - same as training
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode(
        self, 
        sentences: List[str], 
        batch_size: int = 32, 
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode sentences using the same method as training.
        Adds 'query: ' prefix for e5-base-v2 compatibility.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            convert_to_numpy: Whether to convert to numpy array
            show_progress_bar: Whether to show progress bar
        
        Returns:
            np.ndarray: Encoded embeddings (shape: [num_sentences, 768])
        """
        self.eval()  # Set model to evaluation mode
        
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Add 'query: ' prefix for e5-base-v2 (required by model)
        # Handle None values and empty strings
        processed_sentences = []
        for sentence in sentences:
            if sentence is None:
                processed_sentences.append("query: ")  # Default empty query
            elif isinstance(sentence, str):
                processed_sentences.append(f"query: {sentence.strip()}")
            else:
                processed_sentences.append(f"query: {str(sentence)}")
        
        all_embeddings = []
        
        # Encode in batches
        with torch.no_grad():  # No gradient computation
            for i in range(0, len(processed_sentences), batch_size):
                batch_sentences = processed_sentences[i:i + batch_size]
                
                # Tokenize batch
                tokens = self.tokenizer(
                    batch_sentences,
                    truncation=True,
                    padding=True,
                    max_length=128,  # Same as training
                    return_tensors='pt'
                ).to(next(self.parameters()).device)
                
                # Get embeddings
                embeddings = self.forward(tokens['input_ids'], tokens['attention_mask'])
                
                # Convert to numpy if requested
                if convert_to_numpy:
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
        
        # Combine all batches
        if convert_to_numpy:
            all_embeddings = np.vstack(all_embeddings)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)
        
        return all_embeddings


# ============================================================================
# CUSTOM RETRIEVER MODEL (Wrapper)
# ============================================================================

class CustomRetrieverModel:
    """
    Wrapper for your custom trained retriever model.
    Handles both knowledge base documents and query encoding.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize retriever model.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Create model instance
        self.model = CustomSentenceTransformer("intfloat/e5-base-v2").to(device)
        
        # Load your trained weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Custom retriever model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            print("🔄 Using base e5-base-v2 model (not trained)...")
        
        # Set to evaluation mode
        self.model.eval()
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode knowledge base documents.
        These are the responses/instructions we're retrieving.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
        
        Returns:
            np.ndarray: Document embeddings (shape: [num_docs, 768])
        """
        return self.model.encode(documents, batch_size=batch_size, convert_to_numpy=True)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode user query for retrieval.
        
        Args:
            query: User query text
        
        Returns:
            np.ndarray: Query embedding (shape: [1, 768])
        """
        return self.model.encode([query], convert_to_numpy=True)


# ============================================================================
# MODULE-LEVEL CACHING (Load once on import)
# ============================================================================

# Global variables for caching
RETRIEVER_MODEL: Optional[CustomRetrieverModel] = None
FAISS_INDEX: Optional[faiss.Index] = None
KB_DATA: Optional[List[Dict]] = None


def load_retriever() -> CustomRetrieverModel:
    """
    Load custom retriever model (called once on startup).
    Uses module-level caching - model stays in RAM.
    
    Returns:
        CustomRetrieverModel: Loaded retriever model
    """
    global RETRIEVER_MODEL
    
    if RETRIEVER_MODEL is None:
        print(f"Loading custom retriever from {settings.RETRIEVER_MODEL_PATH}...")
        RETRIEVER_MODEL = CustomRetrieverModel(
            model_path=settings.RETRIEVER_MODEL_PATH,
            device=settings.DEVICE
        )
        print("✅ Retriever model loaded and cached")
    
    return RETRIEVER_MODEL


def load_faiss_index():
    """
    Load FAISS index + knowledge base from pickle file.
    Uses module-level caching - loaded once on startup.
    
    Returns:
        tuple: (faiss.Index, List[Dict]) - FAISS index and KB data
    """
    global FAISS_INDEX, KB_DATA
    
    if FAISS_INDEX is None or KB_DATA is None:
        print(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}...")
        
        try:
            # Load pickled FAISS index + KB data
            with open(settings.FAISS_INDEX_PATH, 'rb') as f:
                FAISS_INDEX, KB_DATA = pickle.load(f)
            
            print(f"✅ FAISS index loaded: {FAISS_INDEX.ntotal} vectors")
            print(f"✅ Knowledge base loaded: {len(KB_DATA)} documents")
        
        except FileNotFoundError:
            print(f"❌ FAISS index file not found: {settings.FAISS_INDEX_PATH}")
            print("⚠️  You need to create the FAISS index first!")
            raise
        
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            raise
    
    return FAISS_INDEX, KB_DATA


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================

def retrieve_documents(
    query: str, 
    top_k: int = None, 
    min_similarity: float = None
) -> List[Dict]:
    """
    Retrieve top-k documents for a query using custom retriever + FAISS.
    
    Args:
        query: User query text
        top_k: Number of documents to retrieve (default from config)
        min_similarity: Minimum similarity threshold (default from config)
    
    Returns:
        List[Dict]: Retrieved documents with scores
            Each dict contains:
            - instruction: FAQ question
            - response: FAQ answer
            - category: Document category
            - intent: Document intent
            - score: Similarity score (0-1)
            - rank: Rank in results (1-indexed)
            - faq_id: Document ID
    """
    # Use config defaults if not provided
    if top_k is None:
        top_k = settings.TOP_K
    if min_similarity is None:
        min_similarity = settings.SIMILARITY_THRESHOLD
    
    # Validate query
    if not query or query.strip() == "":
        print("⚠️ Empty query provided")
        return []
    
    # Load models (cached, no overhead after first call)
    retriever = load_retriever()
    index, kb = load_faiss_index()
    
    try:
        # Step 1: Encode query
        query_embedding = retriever.encode_query(query)
        
        # Step 2: Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Step 3: Search in FAISS index
        similarities, indices = index.search(query_embedding, top_k)
        
        # Step 4: Check similarity threshold for top result
        if similarities[0][0] < min_similarity:
            print(f"🚫 NO_FETCH (similarity: {similarities[0][0]:.3f} < {min_similarity})")
            return []
        
        print(f"✅ FETCH (similarity: {similarities[0][0]:.3f} >= {min_similarity})")
        
        # Step 5: Format results
        results = []
        for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(kb):
                doc = kb[idx]
                results.append({
                    'instruction': doc.get('instruction', ''),
                    'response': doc.get('response', ''),
                    'category': doc.get('category', 'Unknown'),
                    'intent': doc.get('intent', 'Unknown'),
                    'score': float(similarity),
                    'rank': rank + 1,
                    'faq_id': doc.get('faq_id', f'doc_{idx}')
                })
        
        return results
    
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_context(retrieved_docs: List[Dict], max_context_length: int = None) -> str:
    """
    Format retrieved documents into context string for LLM.
    Prioritizes by score and limits total length.
    
    Args:
        retrieved_docs: List of retrieved documents
        max_context_length: Maximum context length in characters
    
    Returns:
        str: Formatted context string
    """
    if max_context_length is None:
        max_context_length = settings.MAX_CONTEXT_LENGTH
    
    if not retrieved_docs:
        return ""
    
    context_parts = []
    current_length = 0
    
    for doc in retrieved_docs:
        # Create context entry with None checks
        instruction = doc.get('instruction', '') or ''
        response = doc.get('response', '') or ''
        category = doc.get('category', 'N/A') or 'N/A'
        
        context_entry = f"[Rank {doc['rank']}, Score: {doc['score']:.3f}]\n"
        context_entry += f"Q: {instruction}\n"
        context_entry += f"A: {response}\n"
        context_entry += f"Category: {category}\n\n"
        
        # Check length limit
        if current_length + len(context_entry) > max_context_length:
            break
        
        context_parts.append(context_entry)
        current_length += len(context_entry)
    
    return "".join(context_parts)


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your service file:

from app.ml.retriever import retrieve_documents, format_context

# Retrieve documents
docs = retrieve_documents("What is my account balance?", top_k=5)

# Format context for LLM
context = format_context(docs)

# Use context in LLM prompt
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
"""
