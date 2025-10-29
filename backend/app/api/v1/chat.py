"""
Chat API Endpoints (WITH AUTHENTICATION)
RESTful API for the Banking RAG Chatbot

NOW REQUIRES JWT TOKEN FOR ALL ENDPOINTS!

Endpoints:
- POST /chat - Send a message and get response (PROTECTED)
- GET /chat/history/{conversation_id} - Get conversation history (PROTECTED)
- POST /chat/conversation - Create new conversation (PROTECTED)
- GET /chat/conversations - List user's conversations (PROTECTED)
- DELETE /chat/conversation/{conversation_id} - Delete conversation (PROTECTED)
- GET /chat/health - Health check (PUBLIC)
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

from app.services.chat_service import chat_service
from app.db.repositories.conversation_repository import ConversationRepository
from app.utils.dependencies import get_current_user  # AUTH DEPENDENCY
from app.models.user import TokenData  # USER DATA FROM TOKEN


# ============================================================================
# CREATE ROUTER
# ============================================================================
router = APIRouter()


# ============================================================================
# DEPENDENCY: Get ConversationRepository instance
# ============================================================================
def get_conversation_repo() -> ConversationRepository:
    """
    Dependency that provides ConversationRepository instance.
    This ensures MongoDB is connected before repository is used.
    """
    return ConversationRepository()


# ============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., description="User query text", min_length=1, max_length=1000)
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is my account balance?",
                "conversation_id": "conv-123"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Generated response text")
    conversation_id: str = Field(..., description="Conversation ID")
    policy_action: str = Field(..., description="Policy decision: FETCH or NO_FETCH")
    policy_confidence: float = Field(..., description="Policy confidence score (0-1)")
    documents_retrieved: int = Field(..., description="Number of documents retrieved")
    top_doc_score: Optional[float] = Field(None, description="Best document similarity score")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp (ISO format)")


class ConversationCreateResponse(BaseModel):
    """Response after creating a conversation"""
    conversation_id: str = Field(..., description="Created conversation ID")
    created_at: str = Field(..., description="Creation timestamp")


class MessageModel(BaseModel):
    """Single message in conversation history"""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")
    metadata: Optional[Dict] = Field(None, description="Optional metadata")


class ConversationHistoryResponse(BaseModel):
    """Response containing conversation history"""
    conversation_id: str
    messages: List[MessageModel]
    message_count: int


# ============================================================================
# ENDPOINTS (ALL PROTECTED WITH JWT)
# ============================================================================

@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user),
    repo: ConversationRepository = Depends(get_conversation_repo)  # ← INJECT REPO
):
    """
    Main chat endpoint - Send a query and get a response.
    
    **REQUIRES AUTHENTICATION** - JWT token must be provided in Authorization header.
    """
    try:
        # Get user_id from token
        user_id = current_user.user_id
        
        # If no conversation_id provided, create a new conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = await repo.create_conversation(user_id=user_id)
        else:
            # Verify user owns this conversation
            conversation = await repo.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            if conversation["user_id"] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied - you don't own this conversation"
                )
        
        # Get conversation history
        history = await repo.get_conversation_history(
            conversation_id=conversation_id,
            max_messages=10
        )
        
        # Save user message
        await repo.add_message(
            conversation_id=conversation_id,
            message={
                'role': 'user',
                'content': request.query,
                'timestamp': datetime.now()
            }
        )
        
        # Process query through RAG pipeline
        result = await chat_service.process_query(
            query=request.query,
            conversation_history=history,
            user_id=user_id
        )
        
        # Save assistant message
        await repo.add_message(
            conversation_id=conversation_id,
            message={
                'role': 'assistant',
                'content': result['response'],
                'timestamp': datetime.now(),
                'metadata': {
                    'policy_action': result['policy_action'],
                    'policy_confidence': result['policy_confidence'],
                    'documents_retrieved': result['documents_retrieved'],
                    'top_doc_score': result['top_doc_score']
                }
            }
        )
        
        # Log retrieval data for RL training
        await repo.log_retrieval({
            'conversation_id': conversation_id,
            'user_id': user_id,
            'query': request.query,
            'policy_action': result['policy_action'],
            'policy_confidence': result['policy_confidence'],
            'should_retrieve': result['should_retrieve'],
            'documents_retrieved': result['documents_retrieved'],
            'top_doc_score': result['top_doc_score'],
            'response': result['response'],
            'retrieval_time_ms': result['retrieval_time_ms'],
            'generation_time_ms': result['generation_time_ms'],
            'total_time_ms': result['total_time_ms'],
            'retrieved_docs_metadata': result.get('retrieved_docs_metadata', []),
            'timestamp': datetime.now()
        })
        
        # Return response
        return ChatResponse(
            response=result['response'],
            conversation_id=conversation_id,
            policy_action=result['policy_action'],
            policy_confidence=result['policy_confidence'],
            documents_retrieved=result['documents_retrieved'],
            top_doc_score=result['top_doc_score'],
            total_time_ms=result['total_time_ms'],
            timestamp=result['timestamp']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )


@router.post("/conversation", response_model=ConversationCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    current_user: TokenData = Depends(get_current_user),
    repo: ConversationRepository = Depends(get_conversation_repo)
):
    """Create a new conversation"""
    try:
        conversation_id = await repo.create_conversation(user_id=current_user.user_id)
        return ConversationCreateResponse(
            conversation_id=conversation_id,
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/history/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
    repo: ConversationRepository = Depends(get_conversation_repo)
):
    """Get conversation history by ID"""
    try:
        conversation = await repo.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        if conversation["user_id"] != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - you don't own this conversation"
            )
        
        messages = []
        for msg in conversation.get('messages', []):
            messages.append(MessageModel(
                role=msg['role'],
                content=msg['content'],
                timestamp=msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp'],
                metadata=msg.get('metadata')
            ))
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            message_count=len(messages)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conversation history: {str(e)}"
        )


@router.get("/conversations")
async def list_user_conversations(
    limit: int = 10,
    skip: int = 0,
    current_user: TokenData = Depends(get_current_user),
    repo: ConversationRepository = Depends(get_conversation_repo)
):
    """List all conversations for the authenticated user"""
    try:
        conversations = await repo.get_user_conversations(
            user_id=current_user.user_id,
            limit=limit,
            skip=skip
        )
        
        return {
            "user_id": current_user.user_id,
            "user_email": current_user.email,
            "conversations": [
                {
                    "conversation_id": conv['conversation_id'],
                    "created_at": conv['created_at'].isoformat() if isinstance(conv['created_at'], datetime) else conv['created_at'],
                    "updated_at": conv['updated_at'].isoformat() if isinstance(conv['updated_at'], datetime) else conv['updated_at'],
                    "message_count": len(conv.get('messages', []))
                }
                for conv in conversations
            ],
            "total": len(conversations)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conversations: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
    repo: ConversationRepository = Depends(get_conversation_repo)
):
    """Delete a conversation"""
    try:
        conversation = await repo.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        if conversation["user_id"] != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - you don't own this conversation"
            )
        
        success = await repo.delete_conversation(conversation_id)
        
        if success:
            return {
                "message": "Conversation deleted successfully",
                "conversation_id": conversation_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete conversation"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@router.get("/health")
async def chat_health():
    """Health check for chat service (PUBLIC)"""
    try:
        health = await chat_service.health_check()
        
        return {
            "status": "healthy",
            "service": "chat",
            "components": health['components'],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "chat",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
# ============================================================================









# """
# Chat API Endpoints (WITH AUTHENTICATION)
# RESTful API for the Banking RAG Chatbot

# NOW REQUIRES JWT TOKEN FOR ALL ENDPOINTS!

# Endpoints:
# - POST /chat - Send a message and get response (PROTECTED)
# - GET /chat/history/{conversation_id} - Get conversation history (PROTECTED)
# - POST /chat/conversation - Create new conversation (PROTECTED)
# - GET /chat/conversations - List user's conversations (PROTECTED)
# - DELETE /chat/conversation/{conversation_id} - Delete conversation (PROTECTED)
# - GET /chat/health - Health check (PUBLIC)
# """

# from fastapi import APIRouter, HTTPException, status, Depends
# from pydantic import BaseModel, Field
# from typing import List, Dict, Optional
# from datetime import datetime

# from app.services.chat_service import chat_service
# from app.db.repositories.conversation_repository import ConversationRepository
# from app.utils.dependencies import get_current_user  # AUTH DEPENDENCY
# from app.models.user import TokenData  # USER DATA FROM TOKEN


# # ============================================================================
# # CREATE ROUTER
# # ============================================================================
# router = APIRouter()

# # Initialize repository
# conversation_repo = ConversationRepository()


# # ============================================================================
# # PYDANTIC MODELS (Request/Response schemas)
# # ============================================================================

# class ChatRequest(BaseModel):
#     """
#     Request model for chat endpoint.
    
#     NOTE: user_id is now extracted from JWT token, not from request body!
    
#     Example:
#         {
#             "query": "What is my account balance?",
#             "conversation_id": "abc-123"
#         }
#     """
#     query: str = Field(..., description="User query text", min_length=1, max_length=1000)
#     conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "query": "What is my account balance?",
#                 "conversation_id": "conv-123"
#             }
#         }


# class ChatResponse(BaseModel):
#     """
#     Response model for chat endpoint.
    
#     Contains the generated response plus metadata about the RAG pipeline.
#     """
#     response: str = Field(..., description="Generated response text")
#     conversation_id: str = Field(..., description="Conversation ID")
#     policy_action: str = Field(..., description="Policy decision: FETCH or NO_FETCH")
#     policy_confidence: float = Field(..., description="Policy confidence score (0-1)")
#     documents_retrieved: int = Field(..., description="Number of documents retrieved")
#     top_doc_score: Optional[float] = Field(None, description="Best document similarity score")
#     total_time_ms: float = Field(..., description="Total processing time in milliseconds")
#     timestamp: str = Field(..., description="Response timestamp (ISO format)")


# class ConversationCreateRequest(BaseModel):
#     """Request to create a new conversation (no user_id needed - from token)"""
#     pass  # Empty - user_id comes from JWT token


# class ConversationCreateResponse(BaseModel):
#     """Response after creating a conversation"""
#     conversation_id: str = Field(..., description="Created conversation ID")
#     created_at: str = Field(..., description="Creation timestamp")


# class MessageModel(BaseModel):
#     """Single message in conversation history"""
#     role: str = Field(..., description="Message role: user or assistant")
#     content: str = Field(..., description="Message content")
#     timestamp: str = Field(..., description="Message timestamp")
#     metadata: Optional[Dict] = Field(None, description="Optional metadata")


# class ConversationHistoryResponse(BaseModel):
#     """Response containing conversation history"""
#     conversation_id: str
#     messages: List[MessageModel]
#     message_count: int


# # ============================================================================
# # ENDPOINTS (ALL PROTECTED WITH JWT)
# # ============================================================================

# @router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
# async def chat(
#     request: ChatRequest,
#     current_user: TokenData = Depends(get_current_user)  # ← REQUIRES AUTH!
# ):
#     """
#     Main chat endpoint - Send a query and get a response.
    
#     **REQUIRES AUTHENTICATION** - JWT token must be provided in Authorization header.
    
#     This endpoint:
#     1. Extracts user_id from JWT token
#     2. Processes the query through the RAG pipeline
#     3. Saves messages to MongoDB
#     4. Logs retrieval data for RL training
#     5. Returns response with metadata
    
#     Args:
#         request: ChatRequest with query and optional conversation_id
#         current_user: Authenticated user data from JWT token
    
#     Returns:
#         ChatResponse: Generated response with metadata
    
#     Raises:
#         HTTPException: If processing fails or user not authenticated
#     """
#     try:
#         # Get user_id from token (NOT from request body!)
#         user_id = current_user.user_id
        
#         # If no conversation_id provided, create a new conversation
#         conversation_id = request.conversation_id
#         if not conversation_id:
#             conversation_id = await conversation_repo.create_conversation(
#                 user_id=user_id
#             )
#         else:
#             # Verify user owns this conversation
#             conversation = await conversation_repo.get_conversation(conversation_id)
#             if not conversation:
#                 raise HTTPException(
#                     status_code=status.HTTP_404_NOT_FOUND,
#                     detail="Conversation not found"
#                 )
#             if conversation["user_id"] != user_id:
#                 raise HTTPException(
#                     status_code=status.HTTP_403_FORBIDDEN,
#                     detail="Access denied - you don't own this conversation"
#                 )
        
#         # Get conversation history
#         history = await conversation_repo.get_conversation_history(
#             conversation_id=conversation_id,
#             max_messages=10  # Last 5 turns (10 messages)
#         )
        
#         # Save user message to database
#         await conversation_repo.add_message(
#             conversation_id=conversation_id,
#             message={
#                 'role': 'user',
#                 'content': request.query,
#                 'timestamp': datetime.now()
#             }
#         )
        
#         # Process query through RAG pipeline
#         result = await chat_service.process_query(
#             query=request.query,
#             conversation_history=history,
#             user_id=user_id
#         )
        
#         # Save assistant message to database
#         await conversation_repo.add_message(
#             conversation_id=conversation_id,
#             message={
#                 'role': 'assistant',
#                 'content': result['response'],
#                 'timestamp': datetime.now(),
#                 'metadata': {
#                     'policy_action': result['policy_action'],
#                     'policy_confidence': result['policy_confidence'],
#                     'documents_retrieved': result['documents_retrieved'],
#                     'top_doc_score': result['top_doc_score']
#                 }
#             }
#         )
        
#         # Log retrieval data for RL training
#         await conversation_repo.log_retrieval({
#             'conversation_id': conversation_id,
#             'user_id': user_id,
#             'query': request.query,
#             'policy_action': result['policy_action'],
#             'policy_confidence': result['policy_confidence'],
#             'should_retrieve': result['should_retrieve'],
#             'documents_retrieved': result['documents_retrieved'],
#             'top_doc_score': result['top_doc_score'],
#             'response': result['response'],
#             'retrieval_time_ms': result['retrieval_time_ms'],
#             'generation_time_ms': result['generation_time_ms'],
#             'total_time_ms': result['total_time_ms'],
#             'retrieved_docs_metadata': result.get('retrieved_docs_metadata', []),
#             'timestamp': datetime.now()
#         })
        
#         # Return response
#         return ChatResponse(
#             response=result['response'],
#             conversation_id=conversation_id,
#             policy_action=result['policy_action'],
#             policy_confidence=result['policy_confidence'],
#             documents_retrieved=result['documents_retrieved'],
#             top_doc_score=result['top_doc_score'],
#             total_time_ms=result['total_time_ms'],
#             timestamp=result['timestamp']
#         )
    
#     except HTTPException:
#         raise  # Re-raise HTTP exceptions
#     except Exception as e:
#         print(f"❌ Chat endpoint error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to process chat request: {str(e)}"
#         )


# @router.post("/conversation", response_model=ConversationCreateResponse, status_code=status.HTTP_201_CREATED)
# async def create_conversation(
#     current_user: TokenData = Depends(get_current_user)  # ← REQUIRES AUTH!
# ):
#     """
#     Create a new conversation.
    
#     **REQUIRES AUTHENTICATION** - User ID is extracted from JWT token.
    
#     Args:
#         current_user: Authenticated user data from JWT token
    
#     Returns:
#         ConversationCreateResponse: Created conversation ID
#     """
#     try:
#         conversation_id = await conversation_repo.create_conversation(
#             user_id=current_user.user_id
#         )
        
#         return ConversationCreateResponse(
#             conversation_id=conversation_id,
#             created_at=datetime.now().isoformat()
#         )
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create conversation: {str(e)}"
#         )


# @router.get("/history/{conversation_id}", response_model=ConversationHistoryResponse)
# async def get_conversation_history(
#     conversation_id: str,
#     current_user: TokenData = Depends(get_current_user)  # ← REQUIRES AUTH!
# ):
#     """
#     Get conversation history by ID.
    
#     **REQUIRES AUTHENTICATION** - User can only access their own conversations.
    
#     Args:
#         conversation_id: Conversation ID
#         current_user: Authenticated user data from JWT token
    
#     Returns:
#         ConversationHistoryResponse: List of messages
        
#     Raises:
#         HTTPException: If conversation not found or user doesn't own it
#     """
#     try:
#         # Get conversation
#         conversation = await conversation_repo.get_conversation(conversation_id)
        
#         if not conversation:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Conversation {conversation_id} not found"
#             )
        
#         # Verify user owns this conversation
#         if conversation["user_id"] != current_user.user_id:
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail="Access denied - you don't own this conversation"
#             )
        
#         # Format messages
#         messages = []
#         for msg in conversation.get('messages', []):
#             messages.append(MessageModel(
#                 role=msg['role'],
#                 content=msg['content'],
#                 timestamp=msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp'],
#                 metadata=msg.get('metadata')
#             ))
        
#         return ConversationHistoryResponse(
#             conversation_id=conversation_id,
#             messages=messages,
#             message_count=len(messages)
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to fetch conversation history: {str(e)}"
#         )


# @router.get("/conversations")
# async def list_user_conversations(
#     limit: int = 10,
#     skip: int = 0,
#     current_user: TokenData = Depends(get_current_user)  # ← REQUIRES AUTH!
# ):
#     """
#     List all conversations for the authenticated user.
    
#     **REQUIRES AUTHENTICATION** - User ID is extracted from JWT token.
    
#     Args:
#         limit: Maximum conversations to return (default: 10)
#         skip: Number to skip for pagination (default: 0)
#         current_user: Authenticated user data from JWT token
    
#     Returns:
#         dict: List of conversations for current user
#     """
#     try:
#         conversations = await conversation_repo.get_user_conversations(
#             user_id=current_user.user_id,  # From JWT token!
#             limit=limit,
#             skip=skip
#         )
        
#         # Format response
#         return {
#             "user_id": current_user.user_id,
#             "user_email": current_user.email,
#             "conversations": [
#                 {
#                     "conversation_id": conv['conversation_id'],
#                     "created_at": conv['created_at'].isoformat() if isinstance(conv['created_at'], datetime) else conv['created_at'],
#                     "updated_at": conv['updated_at'].isoformat() if isinstance(conv['updated_at'], datetime) else conv['updated_at'],
#                     "message_count": len(conv.get('messages', []))
#                 }
#                 for conv in conversations
#             ],
#             "total": len(conversations)
#         }
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to fetch conversations: {str(e)}"
#         )


# @router.delete("/conversation/{conversation_id}")
# async def delete_conversation(
#     conversation_id: str,
#     current_user: TokenData = Depends(get_current_user)  # ← REQUIRES AUTH!
# ):
#     """
#     Delete a conversation.
    
#     **REQUIRES AUTHENTICATION** - User can only delete their own conversations.
    
#     Args:
#         conversation_id: Conversation ID to delete
#         current_user: Authenticated user data from JWT token
    
#     Returns:
#         dict: Success message
        
#     Raises:
#         HTTPException: If conversation not found or user doesn't own it
#     """
#     try:
#         # Get conversation
#         conversation = await conversation_repo.get_conversation(conversation_id)
        
#         if not conversation:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Conversation {conversation_id} not found"
#             )
        
#         # Verify user owns this conversation
#         if conversation["user_id"] != current_user.user_id:
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail="Access denied - you don't own this conversation"
#             )
        
#         # Delete conversation
#         success = await conversation_repo.delete_conversation(conversation_id)
        
#         if success:
#             return {
#                 "message": "Conversation deleted successfully",
#                 "conversation_id": conversation_id
#             }
#         else:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to delete conversation"
#             )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to delete conversation: {str(e)}"
#         )


# @router.get("/health")
# async def chat_health():
#     """
#     Health check for chat service.
    
#     **PUBLIC ENDPOINT** - No authentication required.
    
#     Returns:
#         dict: Health status of chat service components
#     """
#     try:
#         health = await chat_service.health_check()
        
#         return {
#             "status": "healthy",
#             "service": "chat",
#             "components": health['components'],
#             "timestamp": datetime.now().isoformat()
#         }
    
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "service": "chat",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }


# # ============================================================================
# # USAGE DOCUMENTATION
# # ============================================================================
# """
# === API USAGE EXAMPLES (WITH AUTHENTICATION) ===

# ALL ENDPOINTS (except /health) NOW REQUIRE JWT TOKEN IN AUTHORIZATION HEADER!

# 1. Register user:
#    POST /api/v1/auth/register
#    Body: {
#        "email": "user@example.com",
#        "password": "SecurePass123",
#        "full_name": "John Doe"
#    }
#    Response: { "access_token": "eyJ...", "user": {...} }

# 2. Login:
#    POST /api/v1/auth/login
#    Body: {
#        "email": "user@example.com",
#        "password": "SecurePass123"
#    }
#    Response: { "access_token": "eyJ...", "user": {...} }

# 3. Send chat message (WITH TOKEN):
#    POST /api/v1/chat/
#    Headers: { "Authorization": "Bearer eyJ..." }
#    Body: {
#        "query": "What is my account balance?",
#        "conversation_id": "conv_abc"  // optional
#    }

# 4. Get conversation history (WITH TOKEN):
#    GET /api/v1/chat/history/conv_abc
#    Headers: { "Authorization": "Bearer eyJ..." }

# 5. List conversations (WITH TOKEN):
#    GET /api/v1/chat/conversations?limit=10
#    Headers: { "Authorization": "Bearer eyJ..." }

# === TESTING WITH CURL ===

# # 1. Register
# TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/register" \
#   -H "Content-Type: application/json" \
#   -d '{"email":"test@test.com","password":"test123","full_name":"Test User"}' \
#   | jq -r '.access_token')

# # 2. Send chat message with token
# curl -X POST "http://localhost:8000/api/v1/chat/" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $TOKEN" \
#   -d '{"query": "What is my balance?"}'
# """
# # ============================================================================

























# # ======================================================================================================
# # OLD CODE
# # ======================================================================================================

# # """
# # Chat API Endpoints
# # RESTful API for the Banking RAG Chatbot

# # Endpoints:
# # - POST /chat - Send a message and get response
# # - GET /chat/history/{conversation_id} - Get conversation history
# # - POST /chat/conversation - Create new conversation
# # - GET /chat/conversations - List user's conversations
# # - GET /chat/health - Health check for chat service
# # """

# # from fastapi import APIRouter, HTTPException, status
# # from pydantic import BaseModel, Field
# # from typing import List, Dict, Optional
# # from datetime import datetime

# # from app.services.chat_service import chat_service
# # from app.db.repositories.conversation_repository import ConversationRepository


# # # ============================================================================
# # # CREATE ROUTER
# # # ============================================================================
# # router = APIRouter()

# # # Initialize repository
# # conversation_repo = ConversationRepository()


# # # ============================================================================
# # # PYDANTIC MODELS (Request/Response schemas)
# # # ============================================================================

# # class ChatRequest(BaseModel):
# #     """
# #     Request model for chat endpoint.
    
# #     Example:
# #         {
# #             "query": "What is my account balance?",
# #             "conversation_id": "abc-123",
# #             "user_id": "user_456"
# #         }
# #     """
# #     query: str = Field(..., description="User query text", min_length=1, max_length=1000)
# #     conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
# #     user_id: str = Field(..., description="User ID")
    
# #     class Config:
# #         json_schema_extra = {
# #             "example": {
# #                 "query": "What is my account balance?",
# #                 "conversation_id": "conv-123",
# #                 "user_id": "user-456"
# #             }
# #         }


# # class ChatResponse(BaseModel):
# #     """
# #     Response model for chat endpoint.
    
# #     Contains the generated response plus metadata about the RAG pipeline.
# #     """
# #     response: str = Field(..., description="Generated response text")
# #     conversation_id: str = Field(..., description="Conversation ID")
# #     policy_action: str = Field(..., description="Policy decision: FETCH or NO_FETCH")
# #     policy_confidence: float = Field(..., description="Policy confidence score (0-1)")
# #     documents_retrieved: int = Field(..., description="Number of documents retrieved")
# #     top_doc_score: Optional[float] = Field(None, description="Best document similarity score")
# #     total_time_ms: float = Field(..., description="Total processing time in milliseconds")
# #     timestamp: str = Field(..., description="Response timestamp (ISO format)")


# # class ConversationCreateRequest(BaseModel):
# #     """Request to create a new conversation"""
# #     user_id: str = Field(..., description="User ID")


# # class ConversationCreateResponse(BaseModel):
# #     """Response after creating a conversation"""
# #     conversation_id: str = Field(..., description="Created conversation ID")
# #     created_at: str = Field(..., description="Creation timestamp")


# # class MessageModel(BaseModel):
# #     """Single message in conversation history"""
# #     role: str = Field(..., description="Message role: user or assistant")
# #     content: str = Field(..., description="Message content")
# #     timestamp: str = Field(..., description="Message timestamp")
# #     metadata: Optional[Dict] = Field(None, description="Optional metadata")


# # class ConversationHistoryResponse(BaseModel):
# #     """Response containing conversation history"""
# #     conversation_id: str
# #     messages: List[MessageModel]
# #     message_count: int


# # # ============================================================================
# # # ENDPOINTS
# # # ============================================================================

# # @router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
# # async def chat(request: ChatRequest):
# #     """
# #     Main chat endpoint - Send a query and get a response.
    
# #     This endpoint:
# #     1. Processes the query through the RAG pipeline
# #     2. Saves messages to MongoDB
# #     3. Logs retrieval data for RL training
# #     4. Returns response with metadata
    
# #     Args:
# #         request: ChatRequest with query, conversation_id, user_id
    
# #     Returns:
# #         ChatResponse: Generated response with metadata
    
# #     Raises:
# #         HTTPException: If processing fails
# #     """
# #     try:
# #         # If no conversation_id provided, create a new conversation
# #         conversation_id = request.conversation_id
# #         if not conversation_id:
# #             conversation_id = await conversation_repo.create_conversation(
# #                 user_id=request.user_id
# #             )
        
# #         # Get conversation history
# #         history = await conversation_repo.get_conversation_history(
# #             conversation_id=conversation_id,
# #             max_messages=10  # Last 5 turns (10 messages)
# #         )
        
# #         # Save user message to database
# #         await conversation_repo.add_message(
# #             conversation_id=conversation_id,
# #             message={
# #                 'role': 'user',
# #                 'content': request.query,
# #                 'timestamp': datetime.now()
# #             }
# #         )
        
# #         # Process query through RAG pipeline
# #         result = await chat_service.process_query(
# #             query=request.query,
# #             conversation_history=history,
# #             user_id=request.user_id
# #         )
        
# #         # Save assistant message to database
# #         await conversation_repo.add_message(
# #             conversation_id=conversation_id,
# #             message={
# #                 'role': 'assistant',
# #                 'content': result['response'],
# #                 'timestamp': datetime.now(),
# #                 'metadata': {
# #                     'policy_action': result['policy_action'],
# #                     'policy_confidence': result['policy_confidence'],
# #                     'documents_retrieved': result['documents_retrieved'],
# #                     'top_doc_score': result['top_doc_score']
# #                 }
# #             }
# #         )
        
# #         # Log retrieval data for RL training
# #         await conversation_repo.log_retrieval({
# #             'conversation_id': conversation_id,
# #             'user_id': request.user_id,
# #             'query': request.query,
# #             'policy_action': result['policy_action'],
# #             'policy_confidence': result['policy_confidence'],
# #             'should_retrieve': result['should_retrieve'],
# #             'documents_retrieved': result['documents_retrieved'],
# #             'top_doc_score': result['top_doc_score'],
# #             'response': result['response'],
# #             'retrieval_time_ms': result['retrieval_time_ms'],
# #             'generation_time_ms': result['generation_time_ms'],
# #             'total_time_ms': result['total_time_ms'],
# #             'retrieved_docs_metadata': result.get('retrieved_docs_metadata', []),
# #             'timestamp': datetime.now()
# #         })
        
# #         # Return response
# #         return ChatResponse(
# #             response=result['response'],
# #             conversation_id=conversation_id,
# #             policy_action=result['policy_action'],
# #             policy_confidence=result['policy_confidence'],
# #             documents_retrieved=result['documents_retrieved'],
# #             top_doc_score=result['top_doc_score'],
# #             total_time_ms=result['total_time_ms'],
# #             timestamp=result['timestamp']
# #         )
    
# #     except Exception as e:
# #         print(f"❌ Chat endpoint error: {e}")
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail=f"Failed to process chat request: {str(e)}"
# #         )


# # @router.post("/conversation", response_model=ConversationCreateResponse, status_code=status.HTTP_201_CREATED)
# # async def create_conversation(request: ConversationCreateRequest):
# #     """
# #     Create a new conversation.
    
# #     Args:
# #         request: ConversationCreateRequest with user_id
    
# #     Returns:
# #         ConversationCreateResponse: Created conversation ID
# #     """
# #     try:
# #         conversation_id = await conversation_repo.create_conversation(
# #             user_id=request.user_id
# #         )
        
# #         return ConversationCreateResponse(
# #             conversation_id=conversation_id,
# #             created_at=datetime.now().isoformat()
# #         )
    
# #     except Exception as e:
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail=f"Failed to create conversation: {str(e)}"
# #         )


# # @router.get("/history/{conversation_id}", response_model=ConversationHistoryResponse)
# # async def get_conversation_history(conversation_id: str):
# #     """
# #     Get conversation history by ID.
    
# #     Args:
# #         conversation_id: Conversation ID
    
# #     Returns:
# #         ConversationHistoryResponse: List of messages
# #     """
# #     try:
# #         # Get conversation
# #         conversation = await conversation_repo.get_conversation(conversation_id)
        
# #         if not conversation:
# #             raise HTTPException(
# #                 status_code=status.HTTP_404_NOT_FOUND,
# #                 detail=f"Conversation {conversation_id} not found"
# #             )
        
# #         # Format messages
# #         messages = []
# #         for msg in conversation.get('messages', []):
# #             messages.append(MessageModel(
# #                 role=msg['role'],
# #                 content=msg['content'],
# #                 timestamp=msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp'],
# #                 metadata=msg.get('metadata')
# #             ))
        
# #         return ConversationHistoryResponse(
# #             conversation_id=conversation_id,
# #             messages=messages,
# #             message_count=len(messages)
# #         )
    
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail=f"Failed to fetch conversation history: {str(e)}"
# #         )


# # @router.get("/conversations")
# # async def list_user_conversations(user_id: str, limit: int = 10, skip: int = 0):
# #     """
# #     List all conversations for a user.
    
# #     Args:
# #         user_id: User ID
# #         limit: Maximum conversations to return (default: 10)
# #         skip: Number to skip for pagination (default: 0)
    
# #     Returns:
# #         dict: List of conversations
# #     """
# #     try:
# #         conversations = await conversation_repo.get_user_conversations(
# #             user_id=user_id,
# #             limit=limit,
# #             skip=skip
# #         )
        
# #         # Format response
# #         return {
# #             "user_id": user_id,
# #             "conversations": [
# #                 {
# #                     "conversation_id": conv['conversation_id'],
# #                     "created_at": conv['created_at'].isoformat() if isinstance(conv['created_at'], datetime) else conv['created_at'],
# #                     "updated_at": conv['updated_at'].isoformat() if isinstance(conv['updated_at'], datetime) else conv['updated_at'],
# #                     "message_count": len(conv.get('messages', []))
# #                 }
# #                 for conv in conversations
# #             ],
# #             "total": len(conversations)
# #         }
    
# #     except Exception as e:
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail=f"Failed to fetch conversations: {str(e)}"
# #         )


# # @router.get("/health")
# # async def chat_health():
# #     """
# #     Health check for chat service.
    
# #     Returns:
# #         dict: Health status of chat service components
# #     """
# #     try:
# #         health = await chat_service.health_check()
        
# #         return {
# #             "status": "healthy",
# #             "service": "chat",
# #             "components": health['components'],
# #             "timestamp": datetime.now().isoformat()
# #         }
    
# #     except Exception as e:
# #         return {
# #             "status": "unhealthy",
# #             "service": "chat",
# #             "error": str(e),
# #             "timestamp": datetime.now().isoformat()
# #         }


# # # ============================================================================
# # # USAGE DOCUMENTATION
# # # ============================================================================
# # """
# # === API USAGE EXAMPLES ===

# # 1. Send a chat message:
# #    POST /api/v1/chat/
# #    Body: {
# #        "query": "What is my account balance?",
# #        "user_id": "user_123",
# #        "conversation_id": "conv_abc"  // optional
# #    }

# # 2. Create new conversation:
# #    POST /api/v1/chat/conversation
# #    Body: {
# #        "user_id": "user_123"
# #    }

# # 3. Get conversation history:
# #    GET /api/v1/chat/history/conv_abc

# # 4. List user's conversations:
# #    GET /api/v1/chat/conversations?user_id=user_123&limit=10&skip=0

# # 5. Check health:
# #    GET /api/v1/chat/health

# # === TESTING WITH CURL ===

# # # Send chat message
# # curl -X POST "http://localhost:8000/api/v1/chat/" \
# #   -H "Content-Type: application/json" \
# #   -d '{
# #     "query": "What is my balance?",
# #     "user_id": "user_123"
# #   }'

# # # Get history
# # curl "http://localhost:8000/api/v1/chat/history/conv_123"

# # === TESTING WITH SWAGGER UI ===

# # After starting the server, visit:
# # http://localhost:8000/docs

# # Interactive API documentation with "Try it out" buttons!
# # """