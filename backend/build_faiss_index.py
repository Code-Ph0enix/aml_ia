"""
Build FAISS Index from Scratch
Creates faiss_index.pkl from your knowledge base and trained retriever model

Run this ONCE before starting the backend:
    python build_faiss_index.py

Author: Banking RAG Chatbot
Date: October 2025
"""


# Add these lines at the very top (after docstring)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

import os
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import List


# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS!
# ============================================================================

# Where is your knowledge base JSONL file?
KB_JSONL_FILE = "data/final_knowledge_base.jsonl"

# Where is your trained retriever model?
RETRIEVER_MODEL_PATH = "models/best_retriever_model.pth"

# Where to save the output FAISS pickle?
OUTPUT_PKL_FILE = "models/faiss_index.pkl"

# Device (auto-detect GPU/CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch size for encoding (reduce if you get OOM errors)
BATCH_SIZE = 32


# ============================================================================
# CUSTOM SENTENCE TRANSFORMER (Same as retriever.py)
# ============================================================================

class CustomSentenceTransformer(nn.Module):
    """
    Custom SentenceTransformer - exact copy from retriever.py
    Uses e5-base-v2 with mean pooling and L2 normalization
    """
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        super().__init__()
        print(f"   Loading base model: {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.encoder.config
        print(f"   ✅ Base model loaded")
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through BERT encoder"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode sentences - same as training"""
        self.eval()
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Add 'query: ' prefix for e5-base-v2
        processed_sentences = [f"query: {s.strip()}" for s in sentences]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(processed_sentences), batch_size):
                batch_sentences = processed_sentences[i:i + batch_size]
                
                # Tokenize
                tokens = self.tokenizer(
                    batch_sentences,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.encoder.device)
                
                # Get embeddings
                embeddings = self.forward(tokens['input_ids'], tokens['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


# ============================================================================
# RETRIEVER MODEL (Wrapper)
# ============================================================================

class RetrieverModel:
    """Wrapper for trained retriever model"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"\n🤖 Loading retriever model...")
        print(f"   Device: {device}")
        
        self.device = device
        self.model = CustomSentenceTransformer("intfloat/e5-base-v2").to(device)
        
        # Load trained weights
        print(f"   Loading weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"   ✅ Trained weights loaded")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load trained weights: {e}")
            print(f"   Using base e5-base-v2 model instead")
        
        self.model.eval()
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents"""
        return self.model.encode(documents, batch_size=batch_size)


# ============================================================================
# MAIN: BUILD FAISS INDEX
# ============================================================================

def build_faiss_index():
    """Main function to build FAISS index from scratch"""
    
    print("=" * 80)
    print("🏗️  BUILDING FAISS INDEX FROM SCRATCH")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: LOAD KNOWLEDGE BASE
    # ========================================================================
    print(f"\n📖 STEP 1: Loading knowledge base...")
    print(f"   File: {KB_JSONL_FILE}")
    
    if not os.path.exists(KB_JSONL_FILE):
        print(f"   ❌ ERROR: File not found!")
        print(f"   Please copy your knowledge base to: {KB_JSONL_FILE}")
        return False
    
    kb_data = []
    with open(KB_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                kb_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"   ✅ Loaded {len(kb_data)} documents")
    
    if len(kb_data) == 0:
        print(f"   ❌ ERROR: Knowledge base is empty!")
        return False
    
    # ========================================================================
    # STEP 2: PREPARE DOCUMENTS FOR ENCODING
    # ========================================================================
    print(f"\n📝 STEP 2: Preparing documents for encoding...")
    
    documents = []
    for i, item in enumerate(kb_data):
        # Combine instruction + response for embedding (same as training)
        instruction = item.get('instruction', '')
        response = item.get('response', '')
        
        # Create combined text
        if instruction and response:
            text = f"{instruction} {response}"
        elif instruction:
            text = instruction
        elif response:
            text = response
        else:
            print(f"   ⚠️  Warning: Document {i} has no content, using placeholder")
            text = "empty document"
        
        documents.append(text)
    
    print(f"   ✅ Prepared {len(documents)} documents for encoding")
    print(f"   Average length: {sum(len(d) for d in documents) / len(documents):.1f} chars")
    
    # ========================================================================
    # STEP 3: LOAD RETRIEVER AND ENCODE DOCUMENTS
    # ========================================================================
    print(f"\n🔮 STEP 3: Encoding documents with trained retriever...")
    
    if not os.path.exists(RETRIEVER_MODEL_PATH):
        print(f"   ❌ ERROR: Retriever model not found!")
        print(f"   Please copy your trained model to: {RETRIEVER_MODEL_PATH}")
        return False
    
    # Load retriever
    retriever = RetrieverModel(RETRIEVER_MODEL_PATH, device=DEVICE)
    
    # Encode all documents
    print(f"   Encoding {len(documents)} documents...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   This may take a few minutes... ☕")
    
    try:
        embeddings = retriever.encode_documents(documents, batch_size=BATCH_SIZE)
        print(f"   ✅ Encoded {embeddings.shape[0]} documents")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
    except Exception as e:
        print(f"   ❌ ERROR during encoding: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # STEP 4: BUILD FAISS INDEX
    # ========================================================================
    print(f"\n🔍 STEP 4: Building FAISS index...")
    
    dimension = embeddings.shape[1]
    print(f"   Dimension: {dimension}")
    
    # Create FAISS index (Inner Product = Cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    print(f"   Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Add to index
    print(f"   Adding {embeddings.shape[0]} vectors to FAISS index...")
    index.add(embeddings.astype('float32'))
    
    print(f"   ✅ FAISS index built successfully")
    print(f"   Total vectors: {index.ntotal}")
    
    # ========================================================================
    # STEP 5: SAVE AS PICKLE FILE
    # ========================================================================
    print(f"\n💾 STEP 5: Saving as pickle file...")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PKL_FILE), exist_ok=True)
    
    # Save tuple of (index, kb_data)
    print(f"   Pickling (index, kb_data) tuple...")
    try:
        with open(OUTPUT_PKL_FILE, 'wb') as f:
            pickle.dump((index, kb_data), f)
        
        file_size_mb = Path(OUTPUT_PKL_FILE).stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved: {OUTPUT_PKL_FILE}")
        print(f"   File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"   ❌ ERROR saving pickle: {e}")
        return False
    
    # ========================================================================
    # STEP 6: VERIFY SAVED FILE
    # ========================================================================
    print(f"\n✅ STEP 6: Verifying saved file...")
    
    try:
        with open(OUTPUT_PKL_FILE, 'rb') as f:
            loaded_index, loaded_kb = pickle.load(f)
        
        print(f"   ✅ Verification successful")
        print(f"   Index vectors: {loaded_index.ntotal}")
        print(f"   KB documents: {len(loaded_kb)}")
        
        if loaded_index.ntotal != len(loaded_kb):
            print(f"   ⚠️  WARNING: Size mismatch detected!")
        
    except Exception as e:
        print(f"   ❌ ERROR verifying file: {e}")
        return False
    
    # ========================================================================
    # SUCCESS!
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎉 SUCCESS! FAISS INDEX BUILT AND SAVED")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Documents: {len(kb_data)}")
    print(f"   Vectors: {index.ntotal}")
    print(f"   Dimension: {dimension}")
    print(f"   File: {OUTPUT_PKL_FILE} ({file_size_mb:.2f} MB)")
    print(f"\n🚀 You can now start the backend:")
    print(f"   cd backend")
    print(f"   uvicorn app.main:app --reload")
    print("=" * 80 + "\n")
    
    return True


# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    success = build_faiss_index()
    
    if not success:
        print("\n" + "=" * 80)
        print("❌ FAILED TO BUILD FAISS INDEX")
        print("=" * 80)
        print("\nPlease check:")
        print("1. Knowledge base file exists: data/final_knowledge_base.jsonl")
        print("2. Retriever model exists: models/best_retriever_model.pth")
        print("3. You have enough RAM (embeddings need ~1GB for 10k docs)")
        print("=" * 80 + "\n")
        exit(1)
