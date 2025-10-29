"""
Multi-LLM Manager for Google Gemini, Groq, and HuggingFace
All three APIs co-exist for different purposes (no fallback logic)

Architecture:
- Google Gemini (Primary): User-facing chat responses (best quality)
- Groq (Secondary): Fast inference for evaluation and specific tasks
- HuggingFace: Model downloads and embeddings (always required)

Each API has its designated purpose based on config settings.
"""

import time
import google.generativeai as genai
from typing import List, Dict, Optional, Literal
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import settings


# ============================================================================
# GOOGLE GEMINI MANAGER
# ============================================================================

class GeminiManager:
    """
    Google Gemini API Manager (Primary LLM)
    Handles Google Pro account with Gemini-1.5-Pro model
    """
    
    def __init__(self):
        """Initialize Gemini API with your Google API key"""
        self.api_key = settings.GOOGLE_API_KEY
        self.model_name = settings.GEMINI_MODEL
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Create model instance with safety settings
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": settings.LLM_TEMPERATURE,
                "max_output_tokens": settings.LLM_MAX_TOKENS,
            }
        )
        
        # Rate limiting tracking
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.last_reset = time.time()
        
        print(f"✅ Gemini Manager initialized: {self.model_name}")
    
    def _check_rate_limits(self):
        """
        Check and reset rate limit counters.
        Gemini Pro: 60 requests/min, 60,000 tokens/min
        """
        current_time = time.time()
        
        # Reset counters every minute
        if current_time - self.last_reset > 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.last_reset = current_time
        
        # Check if limits exceeded
        if self.requests_this_minute >= settings.GEMINI_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.last_reset)
            print(f"⚠️ Gemini rate limit hit. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            self._check_rate_limits()  # Recursive check after waiting
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response using Gemini.
        
        Args:
            messages: List of conversation messages
                Format: [{'role': 'user'/'assistant', 'content': '...'}]
            system_prompt: Optional system prompt (prepended to first message)
        
        Returns:
            str: Generated response text
        """
        self._check_rate_limits()
        
        try:
            # Format messages for Gemini
            # Gemini uses 'user' and 'model' roles
            formatted_messages = []
            
            # Add system prompt as first user message if provided
            if system_prompt:
                formatted_messages.append({
                    'role': 'user',
                    'parts': [system_prompt]
                })
            
            # Convert messages
            for msg in messages:
                role = 'model' if msg['role'] == 'assistant' else 'user'
                formatted_messages.append({
                    'role': role,
                    'parts': [msg['content']]
                })
            
            # Generate response
            chat = self.model.start_chat(history=formatted_messages[:-1])
            response = chat.send_message(formatted_messages[-1]['parts'][0])
            
            # Track rate limits
            self.requests_this_minute += 1
            # Note: Token counting would require additional API call
            # For now, estimate ~4 chars per token
            estimated_tokens = len(response.text) // 4
            self.tokens_this_minute += estimated_tokens
            
            return response.text
        
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            raise


# ============================================================================
# GROQ MANAGER
# ============================================================================

class GroqManager:
    """
    Groq API Manager (Secondary LLM)
    Handles fast inference with Llama-3-70B
    """
    
    def __init__(self):
        """Initialize Groq API with single API key"""
        self.api_key = settings.GROQ_API_KEY
        self.model_name = settings.GROQ_MODEL
        
        # Create ChatGroq instance
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )
        
        # Rate limiting tracking
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.last_reset = time.time()
        
        print(f"✅ Groq Manager initialized: {self.model_name}")
    
    def _check_rate_limits(self):
        """
        Check and reset rate limit counters.
        Groq Free: 30 requests/min, 30,000 tokens/min
        """
        current_time = time.time()
        
        # Reset counters every minute
        if current_time - self.last_reset > 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.last_reset = current_time
        
        # Check if limits exceeded
        if self.requests_this_minute >= settings.GROQ_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.last_reset)
            print(f"⚠️ Groq rate limit hit. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            self._check_rate_limits()
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response using Groq.
        
        Args:
            messages: List of conversation messages
                Format: [{'role': 'user'/'assistant', 'content': '...'}]
            system_prompt: Optional system prompt
        
        Returns:
            str: Generated response text
        """
        self._check_rate_limits()
        
        try:
            # Format messages for LangChain
            formatted_messages = []
            
            # Add system message if provided
            if system_prompt:
                formatted_messages.append(SystemMessage(content=system_prompt))
            
            # Convert conversation messages
            for msg in messages:
                if msg['role'] == 'user':
                    formatted_messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    formatted_messages.append(AIMessage(content=msg['content']))
            
            # Generate response
            response = await self.llm.ainvoke(formatted_messages)
            
            # Track rate limits
            self.requests_this_minute += 1
            # Estimate tokens (rough approximation)
            estimated_tokens = len(response.content) // 4
            self.tokens_this_minute += estimated_tokens
            
            return response.content
        
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            raise


# ============================================================================
# UNIFIED LLM MANAGER (Routes to appropriate LLM)
# ============================================================================

class LLMManager:
    """
    Unified LLM Manager that routes requests to appropriate LLM.
    
    Routing strategy (from config):
    - Chat responses → Gemini (best quality for users)
    - Evaluation → Groq (fast, good enough for RL)
    - Policy → Local BERT (no API call)
    """
    
    def __init__(self):
        """Initialize all LLM managers"""
        self.gemini = None
        self.groq = None
        
        # Initialize Gemini if configured
        if settings.is_gemini_enabled():
            try:
                self.gemini = GeminiManager()
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
        
        # Initialize Groq if configured
        if settings.is_groq_enabled():
            try:
                self.groq = GroqManager()
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq: {e}")
        
        print("✅ LLM Manager initialized")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        task: Literal["chat", "evaluation"] = "chat"
    ) -> str:
        """
        Generate response using appropriate LLM based on task.
        
        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            task: Task type - "chat" (user-facing) or "evaluation" (RL training)
        
        Returns:
            str: Generated response
        
        Raises:
            ValueError: If appropriate LLM is not configured
        """
        # Determine which LLM to use based on task
        llm_choice = settings.get_llm_for_task(task)
        
        if llm_choice == "gemini":
            if self.gemini is None:
                raise ValueError("Gemini API not configured. Set GOOGLE_API_KEY in .env")
            return await self.gemini.generate(messages, system_prompt)
        
        elif llm_choice == "groq":
            if self.groq is None:
                raise ValueError("Groq API not configured. Set GROQ_API_KEY in .env")
            return await self.groq.generate(messages, system_prompt)
        
        else:
            raise ValueError(f"Unknown LLM choice: {llm_choice}")
    
    # async def generate_chat_response(
    #     self,
    #     query: str,
    #     context: str,
    #     history: List[Dict[str, str]]
    # ) -> str:
    #     """
    #     Generate chat response (uses Gemini by default).
        
    #     Args:
    #         query: User query
    #         context: Retrieved context (from FAISS)
    #         history: Conversation history
        
    #     Returns:
    #         str: Chat response
    #     """
    #     # Build system prompt
    #     system_prompt = settings.SYSTEM_PROMPT
    #     if context:
    #         system_prompt += f"\n\nRelevant Information:\n{context}"
        
    #     # Build messages
    #     messages = history + [{'role': 'user', 'content': query}]
        
    #     # Generate using chat LLM (Gemini)
    #     return await self.generate(messages, system_prompt, task="chat")
    
    async def generate_chat_response(
        self,
        query: str,
        context: str,
        history: List[Dict[str, str]]
    ) -> str:
        """Generate chat response (uses Gemini by default)."""
    
        # Import the detailed prompt
        from app.services.chat_service import BANKING_SYSTEM_PROMPT
    
        # Build enhanced system prompt with context
        system_prompt = BANKING_SYSTEM_PROMPT
    
        if context:
            system_prompt += f"\n\nRelevant Knowledge Base Context:\n{context}"
        else:
            system_prompt += "\n\nNo specific banking documents were retrieved for this query. Provide a helpful general response while acknowledging your banking specialization."
    
        # Build messages
        messages = history + [{'role': 'user', 'content': query}]
    
        # Generate using chat LLM (Gemini)
        return await self.generate(messages, system_prompt, task="chat")

    
    
    
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        context: str = ""
    ) -> Dict:
        """
        Evaluate response quality (uses Groq for speed).
        Used during RL training.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context (if any)
        
        Returns:
            dict: Evaluation results
                {'quality': 'Good'/'Bad', 'explanation': '...'}
        """
        eval_prompt = f"""Evaluate this response:
Query: {query}
Response: {response}
Context used: {context if context else 'None'}

Is this response Good or Bad? Respond with just "Good" or "Bad" and brief explanation."""
        
        messages = [{'role': 'user', 'content': eval_prompt}]
        
        # Generate using evaluation LLM (Groq)
        result = await self.generate(messages, task="evaluation")
        
        # Parse result
        quality = "Good" if "Good" in result else "Bad"
        
        return {
            'quality': quality,
            'explanation': result
        }


# ============================================================================
# GLOBAL LLM MANAGER INSTANCE
# ============================================================================
llm_manager = LLMManager()


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your service file:

from app.core.llm_manager import llm_manager

# Generate chat response (uses Gemini)
response = await llm_manager.generate_chat_response(
    query="What is my account balance?",
    context="Your balance is $1000",
    history=[]
)

# Evaluate response (uses Groq)
evaluation = await llm_manager.evaluate_response(
    query="What is my balance?",
    response="Your balance is $1000",
    context="Balance: $1000"
)
"""
