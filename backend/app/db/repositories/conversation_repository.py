# """
# Conversation Repository - MongoDB CRUD operations
# Handles storing and retrieving conversations from MongoDB Atlas

# Repository Pattern: Separates database logic from business logic
# This makes code cleaner and easier to test
# """

# import uuid
# from datetime import datetime
# from typing import List, Dict, Optional
# from bson import ObjectId

# from app.db.mongodb import get_database


# # ============================================================================
# # CONVERSATION REPOSITORY
# # ============================================================================

# class ConversationRepository:
#     """
#     Repository for conversation data in MongoDB.
    
#     Collections used:
#     - conversations: Stores complete conversations with messages
#     - retrieval_logs: Logs each retrieval operation (for RL training)
#     """
    
#     def __init__(self):
#         """Initialize repository with database connection"""
#         self.db = get_database()
#         self.conversations = self.db["conversations"]
#         self.retrieval_logs = self.db["retrieval_logs"]
    
#     # ========================================================================
#     # CONVERSATION CRUD OPERATIONS
#     # ========================================================================
    
#     async def create_conversation(
#         self,
#         user_id: str,
#         conversation_id: Optional[str] = None
#     ) -> str:
#         """
#         Create a new conversation.
        
#         Args:
#             user_id: User ID who owns this conversation
#             conversation_id: Optional custom conversation ID (auto-generated if None)
        
#         Returns:
#             str: Conversation ID
#         """
#         if conversation_id is None:
#             conversation_id = str(uuid.uuid4())
        
#         conversation = {
#             "conversation_id": conversation_id,
#             "user_id": user_id,
#             "messages": [],  # Will store all messages
#             "created_at": datetime.now(),
#             "updated_at": datetime.now(),
#             "status": "active"  # active, archived, deleted
#         }
        
#         await self.conversations.insert_one(conversation)
        
#         return conversation_id
    
#     async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
#         """
#         Get a conversation by ID.
        
#         Args:
#             conversation_id: Conversation ID
        
#         Returns:
#             dict or None: Conversation document
#         """
#         conversation = await self.conversations.find_one(
#             {"conversation_id": conversation_id}
#         )
        
#         # Convert MongoDB ObjectId to string for JSON serialization
#         if conversation and "_id" in conversation:
#             conversation["_id"] = str(conversation["_id"])
        
#         return conversation
    
#     async def get_user_conversations(
#         self,
#         user_id: str,
#         limit: int = 10,
#         skip: int = 0
#     ) -> List[Dict]:
#         """
#         Get all conversations for a user.
        
#         Args:
#             user_id: User ID
#             limit: Maximum number of conversations to return
#             skip: Number of conversations to skip (for pagination)
        
#         Returns:
#             list: List of conversation documents
#         """
#         cursor = self.conversations.find(
#             {"user_id": user_id, "status": "active"}
#         ).sort("updated_at", -1).skip(skip).limit(limit)
        
#         conversations = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds to strings
#         for conv in conversations:
#             if "_id" in conv:
#                 conv["_id"] = str(conv["_id"])
        
#         return conversations
    
#     async def add_message(
#         self,
#         conversation_id: str,
#         message: Dict
#     ) -> bool:
#         """
#         Add a message to a conversation.
        
#         Args:
#             conversation_id: Conversation ID
#             message: Message dict
#                 {
#                     'role': 'user' or 'assistant',
#                     'content': str,
#                     'timestamp': datetime,
#                     'metadata': dict (optional - policy_action, docs_retrieved, etc.)
#                 }
        
#         Returns:
#             bool: Success status
#         """
#         # Ensure timestamp exists
#         if "timestamp" not in message:
#             message["timestamp"] = datetime.now()
        
#         # Add message to conversation
#         result = await self.conversations.update_one(
#             {"conversation_id": conversation_id},
#             {
#                 "$push": {"messages": message},
#                 "$set": {"updated_at": datetime.now()}
#             }
#         )
        
#         return result.modified_count > 0
    
#     async def get_conversation_history(
#         self,
#         conversation_id: str,
#         max_messages: int = None
#     ) -> List[Dict]:
#         """
#         Get conversation history (messages only).
        
#         Args:
#             conversation_id: Conversation ID
#             max_messages: Optional limit on number of messages
        
#         Returns:
#             list: List of messages
#         """
#         conversation = await self.get_conversation(conversation_id)
        
#         if not conversation:
#             return []
        
#         messages = conversation.get("messages", [])
        
#         if max_messages:
#             messages = messages[-max_messages:]
        
#         return messages
    
#     async def delete_conversation(self, conversation_id: str) -> bool:
#         """
#         Soft delete a conversation (mark as deleted, don't actually delete).
        
#         Args:
#             conversation_id: Conversation ID
        
#         Returns:
#             bool: Success status
#         """
#         result = await self.conversations.update_one(
#             {"conversation_id": conversation_id},
#             {
#                 "$set": {
#                     "status": "deleted",
#                     "deleted_at": datetime.now()
#                 }
#             }
#         )
        
#         return result.modified_count > 0
    
#     # ========================================================================
#     # RETRIEVAL LOGS (for RL training)
#     # ========================================================================
    
#     async def log_retrieval(
#         self,
#         log_data: Dict
#     ) -> str:
#         """
#         Log a retrieval operation (for RL training and analysis).
        
#         Args:
#             log_data: Log data dict
#                 {
#                     'conversation_id': str,
#                     'user_id': str,
#                     'query': str,
#                     'policy_action': 'FETCH' or 'NO_FETCH',
#                     'policy_confidence': float,
#                     'documents_retrieved': int,
#                     'top_doc_score': float or None,
#                     'retrieved_docs_metadata': list,
#                     'response': str,
#                     'retrieval_time_ms': float,
#                     'generation_time_ms': float,
#                     'total_time_ms': float,
#                     'timestamp': datetime
#                 }
        
#         Returns:
#             str: Log ID
#         """
#         # Add timestamp if not present
#         if "timestamp" not in log_data:
#             log_data["timestamp"] = datetime.now()
        
#         # Generate log ID
#         log_id = str(uuid.uuid4())
#         log_data["log_id"] = log_id
        
#         # Insert log
#         await self.retrieval_logs.insert_one(log_data)
        
#         return log_id
    
#     async def get_retrieval_logs(
#         self,
#         conversation_id: Optional[str] = None,
#         user_id: Optional[str] = None,
#         limit: int = 100,
#         skip: int = 0
#     ) -> List[Dict]:
#         """
#         Get retrieval logs (for analysis and RL training).
        
#         Args:
#             conversation_id: Optional filter by conversation
#             user_id: Optional filter by user
#             limit: Maximum number of logs
#             skip: Number of logs to skip
        
#         Returns:
#             list: List of log documents
#         """
#         # Build query
#         query = {}
#         if conversation_id:
#             query["conversation_id"] = conversation_id
#         if user_id:
#             query["user_id"] = user_id
        
#         # Fetch logs
#         cursor = self.retrieval_logs.find(query).sort("timestamp", -1).skip(skip).limit(limit)
#         logs = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds to strings
#         for log in logs:
#             if "_id" in log:
#                 log["_id"] = str(log["_id"])
        
#         return logs
    
#     async def get_logs_for_rl_training(
#         self,
#         min_date: Optional[datetime] = None,
#         limit: int = 1000
#     ) -> List[Dict]:
#         """
#         Get logs specifically for RL training.
#         Filters for logs with both policy decision and retrieval results.
        
#         Args:
#             min_date: Optional minimum date for logs
#             limit: Maximum number of logs
        
#         Returns:
#             list: List of log documents suitable for RL training
#         """
#         # Build query
#         query = {
#             "policy_action": {"$exists": True},
#             "response": {"$exists": True}
#         }
        
#         if min_date:
#             query["timestamp"] = {"$gte": min_date}
        
#         # Fetch logs
#         cursor = self.retrieval_logs.find(query).sort("timestamp", -1).limit(limit)
#         logs = await cursor.to_list(length=limit)
        
#         # Convert ObjectIds
#         for log in logs:
#             if "_id" in log:
#                 log["_id"] = str(log["_id"])
        
#         return logs
    
#     # ========================================================================
#     # ANALYTICS QUERIES
#     # ========================================================================
    
#     async def get_conversation_stats(self, user_id: str) -> Dict:
#         """
#         Get conversation statistics for a user.
        
#         Args:
#             user_id: User ID
        
#         Returns:
#             dict: Statistics
#         """
#         # Count total conversations
#         total_conversations = await self.conversations.count_documents({
#             "user_id": user_id,
#             "status": "active"
#         })
        
#         # Count total messages
#         pipeline = [
#             {"$match": {"user_id": user_id, "status": "active"}},
#             {"$project": {"message_count": {"$size": "$messages"}}}
#         ]
        
#         result = await self.conversations.aggregate(pipeline).to_list(length=None)
#         total_messages = sum(doc.get("message_count", 0) for doc in result)
        
#         return {
#             "total_conversations": total_conversations,
#             "total_messages": total_messages,
#             "avg_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0
#         }
    
#     async def get_policy_stats(self, user_id: Optional[str] = None) -> Dict:
#         """
#         Get policy decision statistics.
        
#         Args:
#             user_id: Optional user ID filter
        
#         Returns:
#             dict: Policy statistics
#         """
#         # Build query
#         query = {}
#         if user_id:
#             query["user_id"] = user_id
        
#         # Count FETCH vs NO_FETCH
#         fetch_count = await self.retrieval_logs.count_documents({
#             **query,
#             "policy_action": "FETCH"
#         })
        
#         no_fetch_count = await self.retrieval_logs.count_documents({
#             **query,
#             "policy_action": "NO_FETCH"
#         })
        
#         total = fetch_count + no_fetch_count
        
#         return {
#             "fetch_count": fetch_count,
#             "no_fetch_count": no_fetch_count,
#             "total": total,
#             "fetch_rate": fetch_count / total if total > 0 else 0,
#             "no_fetch_rate": no_fetch_count / total if total > 0 else 0
#         }


# # ============================================================================
# # USAGE EXAMPLE (for reference)
# # ============================================================================
# """
# # In your service or API endpoint:

# from app.db.repositories.conversation_repository import ConversationRepository

# repo = ConversationRepository()

# # Create conversation
# conv_id = await repo.create_conversation(user_id="user_123")

# # Add user message
# await repo.add_message(conv_id, {
#     'role': 'user',
#     'content': 'What is my balance?',
#     'timestamp': datetime.now()
# })

# # Add assistant message
# await repo.add_message(conv_id, {
#     'role': 'assistant',
#     'content': 'Your balance is $1000',
#     'timestamp': datetime.now(),
#     'metadata': {
#         'policy_action': 'FETCH',
#         'documents_retrieved': 3
#     }
# })

# # Get conversation history
# history = await repo.get_conversation_history(conv_id)

# # Log retrieval for RL training
# await repo.log_retrieval({
#     'conversation_id': conv_id,
#     'user_id': 'user_123',
#     'query': 'What is my balance?',
#     'policy_action': 'FETCH',
#     'documents_retrieved': 3,
#     'response': 'Your balance is $1000'
# })
# """





























































"""
Conversation Repository - MongoDB CRUD operations
Handles storing and retrieving conversations from MongoDB Atlas

Repository Pattern: Separates database logic from business logic
This makes code cleaner and easier to test

Collections:
- conversations: Stores complete conversations with messages
- retrieval_logs: Logs each retrieval operation (for RL training data)
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional
from bson import ObjectId

from app.db.mongodb import get_database


# ============================================================================
# CONVERSATION REPOSITORY
# ============================================================================

class ConversationRepository:
    """
    Repository for conversation data in MongoDB.
    
    Provides CRUD operations for:
    1. Conversations (user chat sessions)
    2. Retrieval logs (for RL training and analytics)
    """
    
    def __init__(self):
        """
        Initialize repository with database connection.
        
        Gracefully handles case where MongoDB is not connected.
        """
        self.db = get_database()
        
        # Graceful handling if MongoDB not connected
        if self.db is None:
            print("⚠️ ConversationRepository: MongoDB not connected")
            print("   Repository will not function until database is connected")
            self.conversations = None
            self.retrieval_logs = None
        else:
            self.conversations = self.db["conversations"]
            self.retrieval_logs = self.db["retrieval_logs"]
            print("✅ ConversationRepository initialized with MongoDB")
    
    def _check_connection(self):
        """
        Check if MongoDB is connected.
        
        Raises:
            RuntimeError: If MongoDB is not connected
        """
        if self.db is None or self.conversations is None:
            raise RuntimeError(
                "MongoDB not connected. Cannot perform database operations. "
                "Check MONGODB_URI in .env file."
            )
    
    # ========================================================================
    # CONVERSATION CRUD OPERATIONS
    # ========================================================================
    
    async def create_conversation(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            user_id: User ID who owns this conversation
            conversation_id: Optional custom conversation ID (auto-generated if None)
        
        Returns:
            str: Conversation ID
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        conversation = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "messages": [],  # Will store all messages
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "status": "active"  # active, archived, deleted
        }
        
        await self.conversations.insert_one(conversation)
        
        return conversation_id
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            dict or None: Conversation document
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        conversation = await self.conversations.find_one(
            {"conversation_id": conversation_id}
        )
        
        # Convert MongoDB ObjectId to string for JSON serialization
        if conversation and "_id" in conversation:
            conversation["_id"] = str(conversation["_id"])
        
        return conversation
    
    # async def get_user_conversations(
    #     self,
    #     user_id: str,
    #     limit: int = 10,
    #     skip: int = 0
    # ) -> List[Dict]:
    #     """
    #     Get all conversations for a user.
        
    #     Args:
    #         user_id: User ID
    #         limit: Maximum number of conversations to return
    #         skip: Number of conversations to skip (for pagination)
        
    #     Returns:
    #         list: List of conversation documents
        
    #     Raises:
    #         RuntimeError: If MongoDB not connected
    #     """
    #     self._check_connection()
        
    #     cursor = self.conversations.find(
    #         {"user_id": user_id, "status": "active"}
    #     ).sort("updated_at", -1).skip(skip).limit(limit)
        
    #     conversations = await cursor.to_list(length=limit)
        
    #     # Convert ObjectIds to strings
    #     for conv in conversations:
    #         if "_id" in conv:
    #             conv["_id"] = str(conv["_id"])
        
    #     return conversations
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 10,
        skip: int = 0
    ) -> List[Dict]:
        """Get all conversations for a user."""
        # Gracefully return empty list if not connected
        if self.db is None or self.conversations is None:
            print("⚠️  MongoDB not connected - returning empty conversations list")
            return []
    
        cursor = self.conversations.find(
            {"user_id": user_id, "status": "active"}
        ).sort("updated_at", -1).skip(skip).limit(limit)
    
        conversations = await cursor.to_list(length=limit)
    
        # Convert ObjectIds to strings
        for conv in conversations:
            if "_id" in conv:
                conv["_id"] = str(conv["_id"])
    
        return conversations

    
    async def add_message(
        self,
        conversation_id: str,
        message: Dict
    ) -> bool:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            message: Message dict
                {
                    'role': 'user' or 'assistant',
                    'content': str,
                    'timestamp': datetime,
                    'metadata': dict (optional - policy_action, docs_retrieved, etc.)
                }
        
        Returns:
            bool: Success status
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        # Ensure timestamp exists
        if "timestamp" not in message:
            message["timestamp"] = datetime.now()
        
        # Add message to conversation
        result = await self.conversations.update_one(
            {"conversation_id": conversation_id},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        return result.modified_count > 0
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        max_messages: int = None
    ) -> List[Dict]:
        """
        Get conversation history (messages only).
        
        Args:
            conversation_id: Conversation ID
            max_messages: Optional limit on number of messages
        
        Returns:
            list: List of messages
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        conversation = await self.get_conversation(conversation_id)
        
        if not conversation:
            return []
        
        messages = conversation.get("messages", [])
        
        if max_messages:
            messages = messages[-max_messages:]
        
        return messages
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Soft delete a conversation (mark as deleted, don't actually delete).
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            bool: Success status
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        result = await self.conversations.update_one(
            {"conversation_id": conversation_id},
            {
                "$set": {
                    "status": "deleted",
                    "deleted_at": datetime.now()
                }
            }
        )
        
        return result.modified_count > 0
    
    # ========================================================================
    # RETRIEVAL LOGS (for RL training)
    # ========================================================================
    
    async def log_retrieval(
        self,
        log_data: Dict
    ) -> str:
        """
        Log a retrieval operation (for RL training and analysis).
        
        Args:
            log_data: Log data dict
                {
                    'conversation_id': str,
                    'user_id': str,
                    'query': str,
                    'policy_action': 'FETCH' or 'NO_FETCH',
                    'policy_confidence': float,
                    'documents_retrieved': int,
                    'top_doc_score': float or None,
                    'retrieved_docs_metadata': list,
                    'response': str,
                    'retrieval_time_ms': float,
                    'generation_time_ms': float,
                    'total_time_ms': float,
                    'timestamp': datetime
                }
        
        Returns:
            str: Log ID
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        # Add timestamp if not present
        if "timestamp" not in log_data:
            log_data["timestamp"] = datetime.now()
        
        # Generate log ID
        log_id = str(uuid.uuid4())
        log_data["log_id"] = log_id
        
        # Insert log
        await self.retrieval_logs.insert_one(log_data)
        
        return log_id
    
    async def get_retrieval_logs(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """
        Get retrieval logs (for analysis and RL training).
        
        Args:
            conversation_id: Optional filter by conversation
            user_id: Optional filter by user
            limit: Maximum number of logs
            skip: Number of logs to skip
        
        Returns:
            list: List of log documents
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        # Build query
        query = {}
        if conversation_id:
            query["conversation_id"] = conversation_id
        if user_id:
            query["user_id"] = user_id
        
        # Fetch logs
        cursor = self.retrieval_logs.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        # Convert ObjectIds to strings
        for log in logs:
            if "_id" in log:
                log["_id"] = str(log["_id"])
        
        return logs
    
    async def get_logs_for_rl_training(
        self,
        min_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get logs specifically for RL training.
        Filters for logs with both policy decision and retrieval results.
        
        Args:
            min_date: Optional minimum date for logs
            limit: Maximum number of logs
        
        Returns:
            list: List of log documents suitable for RL training
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        # Build query
        query = {
            "policy_action": {"$exists": True},
            "response": {"$exists": True}
        }
        
        if min_date:
            query["timestamp"] = {"$gte": min_date}
        
        # Fetch logs
        cursor = self.retrieval_logs.find(query).sort("timestamp", -1).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        # Convert ObjectIds
        for log in logs:
            if "_id" in log:
                log["_id"] = str(log["_id"])
        
        return logs
    
    # ========================================================================
    # ANALYTICS QUERIES
    # ========================================================================
    
    async def get_conversation_stats(self, user_id: str) -> Dict:
        """
        Get conversation statistics for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            dict: Statistics
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        # Count total conversations
        total_conversations = await self.conversations.count_documents({
            "user_id": user_id,
            "status": "active"
        })
        
        # Count total messages
        pipeline = [
            {"$match": {"user_id": user_id, "status": "active"}},
            {"$project": {"message_count": {"$size": "$messages"}}}
        ]
        
        result = await self.conversations.aggregate(pipeline).to_list(length=None)
        total_messages = sum(doc.get("message_count", 0) for doc in result)
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "avg_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0
        }
    
    async def get_policy_stats(self, user_id: Optional[str] = None) -> Dict:
        """
        Get policy decision statistics.
        
        Args:
            user_id: Optional user ID filter
        
        Returns:
            dict: Policy statistics
        
        Raises:
            RuntimeError: If MongoDB not connected
        """
        self._check_connection()
        
        # Build query
        query = {}
        if user_id:
            query["user_id"] = user_id
        
        # Count FETCH vs NO_FETCH
        fetch_count = await self.retrieval_logs.count_documents({
            **query,
            "policy_action": "FETCH"
        })
        
        no_fetch_count = await self.retrieval_logs.count_documents({
            **query,
            "policy_action": "NO_FETCH"
        })
        
        total = fetch_count + no_fetch_count
        
        return {
            "fetch_count": fetch_count,
            "no_fetch_count": no_fetch_count,
            "total": total,
            "fetch_rate": fetch_count / total if total > 0 else 0,
            "no_fetch_rate": no_fetch_count / total if total > 0 else 0
        }


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your service or API endpoint:

from app.db.repositories.conversation_repository import ConversationRepository

repo = ConversationRepository()

# Create conversation
conv_id = await repo.create_conversation(user_id="user_123")

# Add user message
await repo.add_message(conv_id, {
    'role': 'user',
    'content': 'What is my balance?',
    'timestamp': datetime.now()
})

# Add assistant message
await repo.add_message(conv_id, {
    'role': 'assistant',
    'content': 'Your balance is $1000',
    'timestamp': datetime.now(),
    'metadata': {
        'policy_action': 'FETCH',
        'documents_retrieved': 3
    }
})

# Get conversation history
history = await repo.get_conversation_history(conv_id)

# Log retrieval for RL training
await repo.log_retrieval({
    'conversation_id': conv_id,
    'user_id': 'user_123',
    'query': 'What is my balance?',
    'policy_action': 'FETCH',
    'documents_retrieved': 3,
    'response': 'Your balance is $1000'
})
"""
