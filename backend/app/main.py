"""
FastAPI Main Application Entry Point
Banking RAG Chatbot API with JWT Authentication

This file:
1. Creates the FastAPI app
2. Configures CORS middleware  
3. Connects to MongoDB on startup/shutdown
4. Includes API routers (auth + chat)
5. Provides health check endpoints
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection


# ============================================================================
# LIFESPAN MANAGER (Startup & Shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.
    
    Startup:
    - Connect to MongoDB Atlas
    - ML models load lazily on first use
    
    Shutdown:
    - Close MongoDB connection
    - Cleanup resources
    """
    # ========================================================================
    # STARTUP
    # ========================================================================
    print("\n" + "=" * 80)
    print("🚀 STARTING BANKING RAG CHATBOT API")
    print("=" * 80)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print("=" * 80)
    
    # Connect to MongoDB
    await connect_to_mongo()
    
    print("\n💡 ML Models Info:")
    print("   Policy Network: Loads on first chat request (lazy loading)")
    print("   Retriever Model: Loads on first retrieval (lazy loading)")
    print("   LLM (Gemini): Connects on first generation")
    
    print("\n✅ Backend startup complete!")
    print("=" * 80)
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"🏥 Health Check: http://localhost:8000/health")
    print(f"🔐 Register: POST http://localhost:8000/api/v1/auth/register")
    print(f"🔑 Login: POST http://localhost:8000/api/v1/auth/login")
    print("=" * 80 + "\n")
    
    yield  # Application runs here
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    print("\n" + "=" * 80)
    print("🛑 SHUTTING DOWN API")
    print("=" * 80)
    
    # Close MongoDB connection
    await close_mongo_connection()
    
    print("✅ Shutdown complete")
    print("=" * 80 + "\n")


# ============================================================================
# CREATE FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Banking RAG Chatbot API",
    description="""
    🤖 AI-powered Banking Assistant with:
    
    **Features:**
    - 🔐 JWT Authentication (Sign up, Login, Protected routes)
    - 💬 RAG (Retrieval-Augmented Generation)
    - 🧠 RL-based Policy Network (BERT)
    - 🔍 Custom E5 Retriever
    - ✨ Google Gemini LLM
    
    **Capabilities:**
    - Intelligent document retrieval
    - Context-aware responses
    - Conversation history
    - Real-time chat
    - User authentication & authorization
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

allowed_origins = settings.get_allowed_origins()

print("\n🌐 CORS Configuration:")
print(f"   Allowed Origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INCLUDE API ROUTERS
# ============================================================================

from app.api.v1 import chat, auth

# Auth router (public endpoints - register, login)
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["🔐 Authentication"]
)

# Chat router (protected endpoints - requires JWT token)
app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["💬 Chat"]
)


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/", tags=["📍 Root"])
async def root():
    """
    Root endpoint - API information and available endpoints
    """
    return {
        "message": "Banking RAG Chatbot API with Authentication",
        "version": "1.0.0",
        "status": "online",
        "authentication": "JWT Bearer Token Required for chat endpoints",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "auth": {
                "register": "POST /api/v1/auth/register",
                "login": "POST /api/v1/auth/login",
                "me": "GET /api/v1/auth/me (requires token)",
                "logout": "POST /api/v1/auth/logout (requires token)"
            },
            "chat": {
                "send_message": "POST /api/v1/chat/ (requires token)",
                "get_history": "GET /api/v1/chat/history/{conversation_id} (requires token)",
                "list_conversations": "GET /api/v1/chat/conversations (requires token)",
                "delete_conversation": "DELETE /api/v1/chat/conversation/{conversation_id} (requires token)"
            },
            "health": "GET /health"
        }
    }


@app.get("/health", tags=["🏥 Health"])
async def health_check():
    """
    Comprehensive health check endpoint
    
    Checks status of:
    - API service
    - MongoDB connection
    - ML models (lazy loaded)
    - Authentication system
    
    Returns:
        dict: Health status of all components
    """
    from app.db.mongodb import get_database
    
    # Check MongoDB
    mongodb_status = "connected" if get_database() is not None else "disconnected"
    
    # Check ML models (don't load them, just check readiness)
    ml_models_status = {
        "policy_network": "ready (lazy load)",
        "retriever": "ready (lazy load)",
        "llm": "ready (API-based)"
    }
    
    # Check authentication
    auth_status = {
        "jwt_enabled": bool(settings.SECRET_KEY and settings.SECRET_KEY != "your-secret-key-change-in-production"),
        "algorithm": settings.ALGORITHM,
        "token_expiry_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES
    }
    
    # Overall health
    is_healthy = mongodb_status == "connected" and auth_status["jwt_enabled"]
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "api": "online",
        "mongodb": mongodb_status,
        "authentication": auth_status,
        "ml_models": ml_models_status,
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG
    }


# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    """
    print(f"\n❌ Unhandled Exception:")
    print(f"   Path: {request.url.path}")
    print(f"   Error: {str(exc)}")
    
    if settings.DEBUG:
        import traceback
        traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )


# ============================================================================
# MAIN ENTRY POINT (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting server directly...")
    print("   Note: For production, use: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG  # Auto-reload only in debug mode
    )
