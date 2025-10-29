"""
Chat Service - Main RAG Pipeline
Combines: Policy Network → Retriever → LLM Generator

This is the core service that orchestrates:
1. Policy decision (FETCH vs NO_FETCH)
2. Document retrieval (if FETCH)
3. Response generation (Gemini)
4. Logging to MongoDB

Adapted from your RAG.py workflow
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.config import settings
from app.ml.policy_network import predict_policy_action
from app.ml.retriever import retrieve_documents, format_context
from app.core.llm_manager import llm_manager

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

BANKING_SYSTEM_PROMPT = """You are an expert banking assistant specialized in Indian financial regulations and banking practices. You have access to a comprehensive knowledge base of banking policies, procedures, and RBI regulations.

Instructions:
- Answer the user query accurately using the provided context when available
- If context is insufficient or query is outside banking domain, still respond helpfully but mention your banking specialization
- If no banking context is available, provide a general helpful response but acknowledge your expertise is in banking
- Never refuse to answer - always be helpful while being transparent about your specialization
- Cite relevant policy numbers or document references when available in context
- Never fabricate specific policies, rates, or eligibility criteria
- If uncertain about current rates or policies, acknowledge the limitation
- Maintain a helpful and professional tone
- Keep responses concise, clear, and actionable
"""

EVALUATION_PROMPT = """You are evaluating a banking assistant's response for quality and accuracy.

Criteria:
1. Accuracy: Is the response factually correct?
2. Relevance: Does it address the user's question?
3. Completeness: Are all aspects of the question covered?
4. Clarity: Is the response easy to understand?
5. Context Usage: Does it properly use the retrieved context?

Rate the response as:
- "Good": Accurate, relevant, complete, and clear
- "Bad": Inaccurate, irrelevant, incomplete, or unclear

Provide your rating and brief explanation."""



# ============================================================================
# CHAT SERVICE
# ============================================================================

class ChatService:
    """
    Main chat service that handles the complete RAG pipeline.
    
    Pipeline:
    1. User query comes in
    2. Policy network decides: FETCH or NO_FETCH
    3. If FETCH: Retrieve documents from FAISS
    4. Generate response using Gemini (with or without context)
    5. Return response + metadata
    """
    
    def __init__(self):
        """Initialize chat service"""
        print("🤖 ChatService initialized")
    
    async def process_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        This is the MAIN function that combines everything:
        - Policy decision
        - Retrieval
        - Generation
        
        Args:
            query: User query text
            conversation_history: Previous conversation turns
                Format: [{'role': 'user'/'assistant', 'content': '...', 'metadata': {...}}]
            user_id: Optional user ID for logging
        
        Returns:
            dict: Complete response with metadata
                {
                    'response': str,                  # Generated response
                    'policy_action': str,             # FETCH or NO_FETCH
                    'policy_confidence': float,       # Confidence score
                    'should_retrieve': bool,          # Whether retrieval was done
                    'documents_retrieved': int,       # Number of docs retrieved
                    'top_doc_score': float or None,   # Best similarity score
                    'retrieval_time_ms': float,       # Time spent on retrieval
                    'generation_time_ms': float,      # Time spent on generation
                    'total_time_ms': float,           # Total processing time
                    'timestamp': str                  # ISO timestamp
                }
        """
        start_time = time.time()
        
        # Initialize history if None
        if conversation_history is None:
            conversation_history = []
        
        # Validate query
        if not query or query.strip() == "":
            return {
                'response': "I didn't receive a valid question. Could you please try again?",
                'policy_action': 'NO_FETCH',
                'policy_confidence': 1.0,
                'should_retrieve': False,
                'documents_retrieved': 0,
                'top_doc_score': None,
                'retrieval_time_ms': 0,
                'generation_time_ms': 0,
                'total_time_ms': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # ====================================================================
        # STEP 1: POLICY DECISION (Local BERT model)
        # ====================================================================
        print(f"\n{'='*80}")
        print(f"🔍 Processing Query: {query[:50]}...")
        print(f"{'='*80}")
        
        policy_start = time.time()
        
        # Predict action using policy network
        policy_result = predict_policy_action(
            query=query,
            history=conversation_history,
            return_probs=True
        )
        
        policy_time = (time.time() - policy_start) * 1000
        
        print(f"\n📊 Policy Decision:")
        print(f"   Action: {policy_result['action']}")
        print(f"   Confidence: {policy_result['confidence']:.3f}")
        print(f"   Should Retrieve: {policy_result['should_retrieve']}")
        print(f"   Time: {policy_time:.2f}ms")
        
        # ====================================================================
        # STEP 2: RETRIEVAL (if FETCH or low confidence NO_FETCH)
        # ====================================================================
        retrieved_docs = []
        context = ""
        retrieval_time = 0
        
        if policy_result['should_retrieve']:
            print(f"\n🔎 Retrieving documents...")
            retrieval_start = time.time()
            
            try:
                # Retrieve documents using custom retriever + FAISS
                retrieved_docs = retrieve_documents(
                    query=query,
                    top_k=settings.TOP_K,
                    min_similarity=settings.SIMILARITY_THRESHOLD
                )
                
                retrieval_time = (time.time() - retrieval_start) * 1000
                
                if retrieved_docs:
                    print(f"   ✅ Retrieved {len(retrieved_docs)} documents")
                    print(f"   Top score: {retrieved_docs[0]['score']:.3f}")
                    
                    # Format context for LLM
                    context = format_context(
                        retrieved_docs,
                        max_context_length=settings.MAX_CONTEXT_LENGTH
                    )
                else:
                    print(f"   ⚠️ No documents above threshold")
            
            except Exception as e:
                print(f"   ❌ Retrieval error: {e}")
                # Continue without retrieval
        
        else:
            print(f"\n🚫 Skipping retrieval (Policy: {policy_result['action']})")
        
        # ====================================================================
        # STEP 3: GENERATE RESPONSE (Gemini)
        # ====================================================================
        print(f"\n💬 Generating response...")
        generation_start = time.time()
        
        try:
            # Generate response using LLM manager (Gemini)
            response = await llm_manager.generate_chat_response(
                query=query,
                context=context,
                history=conversation_history
            )
            
            generation_time = (time.time() - generation_start) * 1000
            
            print(f"   ✅ Response generated")
            print(f"   Length: {len(response)} chars")
            print(f"   Time: {generation_time:.2f}ms")
        
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            response = "I apologize, but I encountered an error generating a response. Please try again."
            generation_time = (time.time() - generation_start) * 1000
        
        # ====================================================================
        # STEP 4: COMPILE RESULTS
        # ====================================================================
        total_time = (time.time() - start_time) * 1000
        
        result = {
            'response': response,
            'policy_action': policy_result['action'],
            'policy_confidence': policy_result['confidence'],
            'should_retrieve': policy_result['should_retrieve'],
            'documents_retrieved': len(retrieved_docs),
            'top_doc_score': retrieved_docs[0]['score'] if retrieved_docs else None,
            'retrieval_time_ms': round(retrieval_time, 2),
            'generation_time_ms': round(generation_time, 2),
            'total_time_ms': round(total_time, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add retrieved docs metadata (for logging, not sent to user)
        if retrieved_docs:
            result['retrieved_docs_metadata'] = [
                {
                    'faq_id': doc['faq_id'],
                    'score': doc['score'],
                    'category': doc['category'],
                    'rank': doc['rank']
                }
                for doc in retrieved_docs
            ]
        
        print(f"\n{'='*80}")
        print(f"✅ Query processed successfully")
        print(f"   Total time: {total_time:.2f}ms")
        print(f"{'='*80}\n")
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all service components.
        
        Returns:
            dict: Health status
        """
        health = {
            'service': 'chat_service',
            'status': 'healthy',
            'components': {}
        }
        
        # Check policy network
        try:
            from app.ml.policy_network import POLICY_MODEL
            health['components']['policy_network'] = 'loaded' if POLICY_MODEL else 'not_loaded'
        except Exception as e:
            health['components']['policy_network'] = f'error: {str(e)}'
        
        # Check retriever
        try:
            from app.ml.retriever import RETRIEVER_MODEL, FAISS_INDEX
            health['components']['retriever'] = 'loaded' if RETRIEVER_MODEL else 'not_loaded'
            health['components']['faiss_index'] = 'loaded' if FAISS_INDEX else 'not_loaded'
        except Exception as e:
            health['components']['retriever'] = f'error: {str(e)}'
        
        # Check LLM manager
        try:
            from app.core.llm_manager import llm_manager as llm
            health['components']['gemini'] = 'enabled' if llm.gemini else 'disabled'
            health['components']['groq'] = 'enabled' if llm.groq else 'disabled'
        except Exception as e:
            health['components']['llm_manager'] = f'error: {str(e)}'
        
        # Overall status
        failed_components = [k for k, v in health['components'].items() if 'error' in str(v)]
        if failed_components:
            health['status'] = 'degraded'
            health['failed_components'] = failed_components
        
        return health


# ============================================================================
# GLOBAL CHAT SERVICE INSTANCE
# ============================================================================
chat_service = ChatService()


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your API endpoint (chat.py):

from app.services.chat_service import chat_service

# Process user query
result = await chat_service.process_query(
    query="What is my account balance?",
    conversation_history=[
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi! How can I help?', 'metadata': {'policy_action': 'NO_FETCH'}}
    ],
    user_id="user_123"
)

# Result contains:
# - response: "Your account balance is $1,234.56"
# - policy_action: "FETCH"
# - documents_retrieved: 3
# - total_time_ms: 450.23
# etc.

# Get service health
health = await chat_service.health_check()
"""
