# LINE 80 VERY IMP CHANGE OF LLM MAX TOKENS FROM 512 TO 1024


"""
Application Configuration
Settings for Banking RAG Chatbot with JWT Authentication
Includes all settings needed by existing llm_manager.py
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # ========================================================================
    # ENVIRONMENT
    # ========================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # ========================================================================
    # MONGODB
    # ========================================================================
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "aml_ia_db")
    
    # ========================================================================
    # JWT AUTHENTICATION
    # ========================================================================
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # ========================================================================
    # CORS (for frontend)
    # ========================================================================
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    
    # ========================================================================
    # GOOGLE GEMINI API
    # ========================================================================
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    GEMINI_REQUESTS_PER_MINUTE: int = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "60"))
    
    # ========================================================================
    # GROQ API (Optional - for evaluation)
    # ========================================================================
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    GROQ_REQUESTS_PER_MINUTE: int = int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "30"))
    
    # ========================================================================
    # HUGGING FACE (Optional - for model downloads)
    # ========================================================================
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # ========================================================================
    # MODEL PATHS (for RL Policy Network and RAG models)
    # ========================================================================
    POLICY_MODEL_PATH: str = os.getenv("POLICY_MODEL_PATH", "app/models/best_policy_model.pth")
    RETRIEVER_MODEL_PATH: str = os.getenv("RETRIEVER_MODEL_PATH", "app/models/best_retriever_model.pth")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "app/models/faiss_index.pkl")
    KB_PATH: str = os.getenv("KB_PATH", "app/data/final_knowledge_base.jsonl")
    
    # ========================================================================
    # DEVICE SETTINGS (for PyTorch/TensorFlow models)
    # ========================================================================
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # ========================================================================
    # LLM PARAMETERS
    # ========================================================================
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024")) # VERY IMPORTANT CHANGE =============================================================================================
    # ============================================================================
    
    # ========================================================================
    # RAG PARAMETERS
    # ========================================================================
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "2000"))
    
    # ========================================================================
    # POLICY NETWORK PARAMETERS
    # ========================================================================
    POLICY_MAX_LEN: int = int(os.getenv("POLICY_MAX_LEN", "256"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # ========================================================================
    # HELPER METHODS (Required by llm_manager.py)
    # ========================================================================
    
    def is_gemini_enabled(self) -> bool:
        """Check if Google Gemini API is configured"""
        return bool(self.GOOGLE_API_KEY and self.GOOGLE_API_KEY != "")
    
    def is_groq_enabled(self) -> bool:
        """Check if Groq API is configured"""
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "")
    
    def is_hf_enabled(self) -> bool:
        """Check if HuggingFace token is configured"""
        return bool(self.HF_TOKEN and self.HF_TOKEN != "")
    
    def get_allowed_origins(self) -> List[str]:
        """Parse allowed origins from comma-separated string"""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    def get_llm_for_task(self, task: str = "qa") -> str:
        """
        Get LLM name for a specific task.
        
        Args:
            task: Task type ('chat', 'evaluation', etc.')
        
        Returns:
            str: LLM name ('gemini' or 'groq')
        """
        # Use Gemini for chat, Groq for evaluation
        if task == "evaluation":
            return "groq" if self.is_groq_enabled() else "gemini"
        else:
            return "gemini"  # Default to Gemini for all tasks


# ============================================================================
# CREATE GLOBAL SETTINGS INSTANCE
# ============================================================================
settings = Settings()


# ============================================================================
# PRINT CONFIGURATION ON LOAD
# ============================================================================
print("=" * 80)
print("✅ Configuration Loaded")
print("=" * 80)
print(f"Environment: {settings.ENVIRONMENT}")
print(f"Debug Mode: {settings.DEBUG}")
print(f"Database: {settings.DATABASE_NAME}")
print(f"Device: {settings.DEVICE}")
print(f"CORS Origins: {settings.ALLOWED_ORIGINS}")
print()
print("🔑 API Keys:")
print(f"   Google Gemini: {'✅ Configured' if settings.is_gemini_enabled() else '❌ Missing'}")
print(f"   Groq API: {'✅ Configured' if settings.is_groq_enabled() else '⚠️  Optional (not set)'}")
print(f"   HuggingFace: {'✅ Configured' if settings.is_hf_enabled() else '⚠️  Optional (not set)'}")
print(f"   MongoDB: {'✅ Configured' if settings.MONGODB_URI else '❌ Missing'}")
print(f"   JWT Secret: {'✅ Configured' if settings.SECRET_KEY != 'your-secret-key-change-in-production' else '⚠️  Using default (CHANGE THIS!)'}")
print()
print("🤖 Model Paths:")
print(f"   Policy Model: {settings.POLICY_MODEL_PATH}")
print(f"   Retriever Model: {settings.RETRIEVER_MODEL_PATH}")
print(f"   FAISS Index: {settings.FAISS_INDEX_PATH}")
print(f"   Knowledge Base: {settings.KB_PATH}")
print("=" * 80)
# ============================================================================

















# """
# Application Configuration
# Settings for Banking RAG Chatbot with JWT Authentication
# Includes all settings needed by existing llm_manager.py
# """

# import os
# from typing import List
# from dotenv import load_dotenv

# load_dotenv()


# class Settings:
#     """Application settings loaded from environment variables"""
    
#     # ========================================================================
#     # ENVIRONMENT
#     # ========================================================================
#     ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
#     DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
#     # ========================================================================
#     # MONGODB
#     # ========================================================================
#     MONGODB_URI: str = os.getenv("MONGODB_URI", "")
#     DATABASE_NAME: str = os.getenv("DATABASE_NAME", "aml_ia_db")
    
#     # ========================================================================
#     # JWT AUTHENTICATION
#     # ========================================================================
#     SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
#     ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
#     # ========================================================================
#     # CORS (for frontend)
#     # ========================================================================
#     ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    
#     # ========================================================================
#     # GOOGLE GEMINI API
#     # ========================================================================
#     GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
#     GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
#     # ========================================================================
#     # GROQ API (Optional - for your llm_manager)
#     # ========================================================================
#     GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
#     GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
#     # ========================================================================
#     # HUGGING FACE (Optional - for model downloads)
#     # ========================================================================
#     HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
#     # ========================================================================
#     # MODEL PATHS (for RL Policy Network and RAG models)
#     # ========================================================================
#     POLICY_MODEL_PATH: str = os.getenv("POLICY_MODEL_PATH", "models/best_policy_model.pth")
#     RETRIEVER_MODEL_PATH: str = os.getenv("RETRIEVER_MODEL_PATH", "models/best_retriever_model.pth")
#     FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "models/faiss_index.pkl")
#     KB_PATH: str = os.getenv("KB_PATH", "data/final_knowledge_base.jsonl")
    
#     # ========================================================================
#     # DEVICE SETTINGS (for PyTorch/TensorFlow models)
#     # ========================================================================
#     DEVICE: str = os.getenv("DEVICE", "cpu")
    
#     # ========================================================================
#     # LLM PARAMETERS
#     # ========================================================================
#     LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
#     LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    
#     # ========================================================================
#     # RAG PARAMETERS
#     # ========================================================================
#     TOP_K: int = int(os.getenv("TOP_K", "5"))
#     SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
#     MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "2000"))
    
#     # ========================================================================
#     # POLICY NETWORK PARAMETERS
#     # ========================================================================
#     POLICY_MAX_LEN: int = int(os.getenv("POLICY_MAX_LEN", "256"))
#     CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

    
#     # ========================================================================
#     # HELPER METHODS (Required by llm_manager.py)
#     # ========================================================================
    
#     def is_gemini_enabled(self) -> bool:
#         """Check if Google Gemini API is configured"""
#         return bool(self.GOOGLE_API_KEY and self.GOOGLE_API_KEY != "")
    
#     def is_groq_enabled(self) -> bool:
#         """Check if Groq API is configured"""
#         return bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "")
    
#     def is_hf_enabled(self) -> bool:
#         """Check if HuggingFace token is configured"""
#         return bool(self.HF_TOKEN and self.HF_TOKEN != "")
    
#     def get_allowed_origins(self) -> List[str]:
#         """Parse allowed origins from comma-separated string"""
#         if self.ALLOWED_ORIGINS == "*":
#             return ["*"]
#         return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
#     # def get_llm_for_task(self, task: str = "qa"):
#     #     """
#     #     Get LLM configuration for a specific task.
#     #     Returns a dict with model settings.
        
#     #     Args:
#     #         task: Task type ('qa', 'retrieval', 'summary', etc.)
        
#     #     Returns:
#     #         dict: LLM configuration
#     #     """
#     #     return {
#     #         'api_key': self.GOOGLE_API_KEY,
#     #         'model': self.GEMINI_MODEL,
#     #         'temperature': self.LLM_TEMPERATURE,
#     #         'max_tokens': self.LLM_MAX_TOKENS,
#     #         'task': task
#     #     }
#     def get_llm_for_task(self, task: str = "qa") -> str:
#         """
#         Get LLM name for a specific task.
    
#         Args:
#             task: Task type ('chat', 'evaluation', etc.)
    
#         Returns:
#             str: LLM name ('gemini' or 'groq')
#         """
#         # Use Gemini for chat, Groq for evaluation
#         if task == "evaluation":
#             return "groq" if self.is_groq_enabled() else "gemini"
#         else:
#             return "gemini"  # Default to Gemini for all other tasks




# # ============================================================================
# # CREATE GLOBAL SETTINGS INSTANCE
# # ============================================================================
# settings = Settings()


# # ============================================================================
# # PRINT CONFIGURATION ON LOAD
# # ============================================================================
# print("=" * 80)
# print("✅ Configuration Loaded")
# print("=" * 80)
# print(f"Environment: {settings.ENVIRONMENT}")
# print(f"Debug Mode: {settings.DEBUG}")
# print(f"Database: {settings.DATABASE_NAME}")
# print(f"Device: {settings.DEVICE}")
# print(f"CORS Origins: {settings.ALLOWED_ORIGINS}")
# print()
# print("🔑 API Keys:")
# print(f"   Google Gemini: {'✅ Configured' if settings.is_gemini_enabled() else '❌ Missing'}")
# print(f"   Groq API: {'✅ Configured' if settings.is_groq_enabled() else '⚠️  Optional (not set)'}")
# print(f"   HuggingFace: {'✅ Configured' if settings.is_hf_enabled() else '⚠️  Optional (not set)'}")
# print(f"   MongoDB: {'✅ Configured' if settings.MONGODB_URI else '❌ Missing'}")
# print(f"   JWT Secret: {'✅ Configured' if settings.SECRET_KEY != 'your-secret-key-change-in-production' else '⚠️  Using default (CHANGE THIS!)'}")
# print()
# print("🤖 Model Paths:")
# print(f"   Policy Model: {settings.POLICY_MODEL_PATH}")
# print(f"   Retriever Model: {settings.RETRIEVER_MODEL_PATH}")
# print(f"   FAISS Index: {settings.FAISS_INDEX_PATH}")
# print(f"   Knowledge Base: {settings.KB_PATH}")
# print("=" * 80)
# # # ============================================================================





















# # """
# # Application Configuration
# # Settings for Banking RAG Chatbot with JWT Authentication
# # Includes all settings needed by existing llm_manager.py
# # """

# # import os
# # from typing import List
# # from dotenv import load_dotenv

# # load_dotenv()


# # class Settings:
# #     """Application settings loaded from environment variables"""
    
# #     # ========================================================================
# #     # ENVIRONMENT
# #     # ========================================================================
# #     ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
# #     DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
# #     # ========================================================================
# #     # MONGODB
# #     # ========================================================================
# #     MONGODB_URI: str = os.getenv("MONGODB_URI", "")
# #     DATABASE_NAME: str = os.getenv("DATABASE_NAME", "aml_ia_db")
    
# #     # ========================================================================
# #     # JWT AUTHENTICATION
# #     # ========================================================================
# #     SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
# #     ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
# #     ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
# #     # ========================================================================
# #     # CORS (for frontend)
# #     # ========================================================================
# #     ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    
# #     # ========================================================================
# #     # GOOGLE GEMINI API
# #     # ========================================================================
# #     GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
# #     GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
# #     # ========================================================================
# #     # GROQ API (Optional - for your llm_manager)
# #     # ========================================================================
# #     GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# #     GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
# #     # ========================================================================
# #     # HUGGING FACE (Optional - for model downloads)
# #     # ========================================================================
# #     HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
# #     # ========================================================================
# #     # HELPER METHODS (Required by llm_manager.py)
# #     # ========================================================================
    
# #     def is_gemini_enabled(self) -> bool:
# #         """Check if Google Gemini API is configured"""
# #         return bool(self.GOOGLE_API_KEY and self.GOOGLE_API_KEY != "")
    
# #     def is_groq_enabled(self) -> bool:
# #         """Check if Groq API is configured"""
# #         return bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "")
    
# #     def is_hf_enabled(self) -> bool:
# #         """Check if HuggingFace token is configured"""
# #         return bool(self.HF_TOKEN and self.HF_TOKEN != "")
    
# #     def get_allowed_origins(self) -> List[str]:
# #         """Parse allowed origins from comma-separated string"""
# #         if self.ALLOWED_ORIGINS == "*":
# #             return ["*"]
# #         return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]


# # # ============================================================================
# # # CREATE GLOBAL SETTINGS INSTANCE
# # # ============================================================================
# # settings = Settings()

# # # ============================================================================
# # # PRINT CONFIGURATION ON LOAD
# # # ============================================================================
# # print("=" * 80)
# # print("✅ Configuration Loaded")
# # print("=" * 80)
# # print(f"Environment: {settings.ENVIRONMENT}")
# # print(f"Debug Mode: {settings.DEBUG}")
# # print(f"Database: {settings.DATABASE_NAME}")
# # # print(f"JWT Algorithm: {settings.ALGORITHM}")
# # # print(f"Token Expiry: {settings.ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
# # print(f"CORS Origins: {settings.ALLOWED_ORIGINS}")
# # print()
# # print("🔑 API Keys:")
# # print(f"   Google Gemini: {'✅ Configured' if settings.is_gemini_enabled() else '❌ Missing'}")
# # print(f"   Groq API: {'✅ Configured' if settings.is_groq_enabled() else '⚠️  Optional (not set)'}")
# # print(f"   HuggingFace: {'✅ Configured' if settings.is_hf_enabled() else '⚠️  Optional (not set)'}")
# # print(f"   MongoDB: {'✅ Configured' if settings.MONGODB_URI else '❌ Missing'}")
# # print(f"   JWT Secret: {'✅ Configured' if settings.SECRET_KEY != 'your-secret-key-change-in-production' else '⚠️  Using default (CHANGE THIS!)'}")
# # print("=" * 80)

























# """
# Application Configuration
# Settings for Banking RAG Chatbot with JWT Authentication
# Includes all settings needed by existing llm_manager.py
# """

# import os
# from typing import List
# from dotenv import load_dotenv

# load_dotenv()


# class Settings:
#     """Application settings loaded from environment variables"""
    
#     # ========================================================================
#     # ENVIRONMENT
#     # ========================================================================
#     ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
#     DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
#     # ========================================================================
#     # MONGODB
#     # ========================================================================
#     MONGODB_URI: str = os.getenv("MONGODB_URI", "")
#     DATABASE_NAME: str = os.getenv("DATABASE_NAME", "aml_ia_db")
    
#     # ========================================================================
#     # JWT AUTHENTICATION
#     # ========================================================================
#     SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
#     ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
#     # ========================================================================
#     # CORS (for frontend)
#     # ========================================================================
#     ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    
#     # ========================================================================
#     # GOOGLE GEMINI API
#     # ========================================================================
#     GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
#     GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
#     # ========================================================================
#     # GROQ API (Optional - for your llm_manager)
#     # ========================================================================
#     GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
#     GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
#     # ========================================================================
#     # HUGGING FACE (Optional - for model downloads)
#     # ========================================================================
#     HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
#     # ========================================================================
#     # MODEL PATHS (for RL Policy Network and RAG models)
#     # ========================================================================
#     POLICY_MODEL_PATH: str = os.getenv("POLICY_MODEL_PATH", "models/best_policy_model.pth")
#     RETRIEVER_MODEL_PATH: str = os.getenv("RETRIEVER_MODEL_PATH", "models/best_retriever_model.pth")
#     FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "models/faiss_index.pkl")
#     KB_PATH: str = os.getenv("KB_PATH", "data/final_knowledge_base.jsonl")
    
#     # ========================================================================
#     # LLM PARAMETERS
#     # ========================================================================
#     LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
#     LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    
#     # ========================================================================
#     # RAG PARAMETERS
#     # ========================================================================
#     TOP_K: int = int(os.getenv("TOP_K", "5"))
#     SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
#     MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "2000"))
    
#     # ========================================================================
#     # HELPER METHODS (Required by llm_manager.py)
#     # ========================================================================
    
#     def is_gemini_enabled(self) -> bool:
#         """Check if Google Gemini API is configured"""
#         return bool(self.GOOGLE_API_KEY and self.GOOGLE_API_KEY != "")
    
#     def is_groq_enabled(self) -> bool:
#         """Check if Groq API is configured"""
#         return bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "")
    
#     def is_hf_enabled(self) -> bool:
#         """Check if HuggingFace token is configured"""
#         return bool(self.HF_TOKEN and self.HF_TOKEN != "")
    
#     def get_allowed_origins(self) -> List[str]:
#         """Parse allowed origins from comma-separated string"""
#         if self.ALLOWED_ORIGINS == "*":
#             return ["*"]
#         return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]


# # ============================================================================
# # CREATE GLOBAL SETTINGS INSTANCE
# # ============================================================================
# settings = Settings()


# # ============================================================================
# # PRINT CONFIGURATION ON LOAD
# # ============================================================================
# print("=" * 80)
# print("✅ Configuration Loaded")
# print("=" * 80)
# print(f"Environment: {settings.ENVIRONMENT}")
# print(f"Debug Mode: {settings.DEBUG}")
# print(f"Database: {settings.DATABASE_NAME}")
# print(f"CORS Origins: {settings.ALLOWED_ORIGINS}")
# print()
# print("🔑 API Keys:")
# print(f"   Google Gemini: {'✅ Configured' if settings.is_gemini_enabled() else '❌ Missing'}")
# print(f"   Groq API: {'✅ Configured' if settings.is_groq_enabled() else '⚠️  Optional (not set)'}")
# print(f"   HuggingFace: {'✅ Configured' if settings.is_hf_enabled() else '⚠️  Optional (not set)'}")
# print(f"   MongoDB: {'✅ Configured' if settings.MONGODB_URI else '❌ Missing'}")
# print(f"   JWT Secret: {'✅ Configured' if settings.SECRET_KEY != 'your-secret-key-change-in-production' else '⚠️  Using default (CHANGE THIS!)'}")
# print("=" * 80)
