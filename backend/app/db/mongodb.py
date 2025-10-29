# """
# MongoDB Connection Handler
# Manages connection to MongoDB Atlas (cloud database)

# This uses Motor - an async MongoDB driver for Python
# Works perfectly with FastAPI's async/await
# """

# from motor.motor_asyncio import AsyncIOMotorClient
# from app.config import settings


# # ============================================================================
# # MONGODB CLIENT SINGLETON
# # ============================================================================
# class MongoDB:
#     """
#     MongoDB client singleton.
#     Stores the connection and database instance.
    
#     Attributes:
#         client: Motor async client connection
#         db: Database instance
#     """
#     client: AsyncIOMotorClient = None
#     db = None


# # Create global instance
# mongodb = MongoDB()


# # ============================================================================
# # CONNECTION FUNCTIONS
# # ============================================================================

# async def connect_to_mongo():
#     """
#     Connect to MongoDB Atlas on application startup.
    
#     This establishes a connection pool that will be reused
#     for all database operations throughout the app's lifetime.
    
#     Connection string format (from .env):
#     mongodb+srv://username:password@cluster.mongodb.net/database
    
#     Raises:
#         Exception: If connection fails
#     """
#     try:
#         # Create async MongoDB client
#         # serverSelectionTimeoutMS: How long to wait before giving up (5 seconds)
#         mongodb.client = AsyncIOMotorClient(
#             settings.MONGODB_URI,
#             serverSelectionTimeoutMS=5000
#         )
        
#         # Get database instance
#         mongodb.db = mongodb.client[settings.DATABASE_NAME]
        
#         # Verify connection by pinging the database
#         await mongodb.client.admin.command('ping')
        
#         print(f"✅ Connected to MongoDB Atlas")
#         print(f"   Database: {settings.DATABASE_NAME}")
        
#     except Exception as e:
#         print(f"❌ MongoDB connection failed: {e}")
#         raise


# async def close_mongo_connection():
#     """
#     Close MongoDB connection on application shutdown.
    
#     This properly closes the connection pool and releases resources.
#     """
#     if mongodb.client:
#         mongodb.client.close()
#         print("✅ MongoDB connection closed")


# def get_database():
#     """
#     Get the current database instance.
    
#     This function is used by repositories to access the database.
    
#     Returns:
#         AsyncIOMotorDatabase: Database instance
        
#     Example:
#         from app.db.mongodb import get_database
        
#         db = get_database()
#         collection = db["conversations"]
#         result = await collection.find_one({"user_id": "123"})
#     """
#     return mongodb.db


# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================

# async def check_connection() -> bool:
#     """
#     Check if MongoDB connection is alive.
    
#     Returns:
#         bool: True if connected, False otherwise
#     """
#     try:
#         if mongodb.client is None:
#             return False
        
#         # Try to ping the database
#         await mongodb.client.admin.command('ping')
#         return True
#     except Exception:
#         return False


# async def get_collection_names():
#     """
#     Get list of all collection names in the database.
#     Useful for debugging and admin operations.
    
#     Returns:
#         list: List of collection names
#     """
#     if mongodb.db is None:
#         return []
    
#     return await mongodb.db.list_collection_names()


# async def create_indexes():
#     """
#     Create database indexes for better query performance.
    
#     This should be called once after first deployment.
#     Indexes speed up queries on specific fields.
    
#     Collections and their indexes:
#     1. conversations:
#        - conversation_id (unique)
#        - user_id (for user queries)
#        - created_at (for sorting)
    
#     2. users:
#        - user_id (unique)
#        - email (unique)
    
#     3. retrieval_logs:
#        - log_id (unique)
#        - timestamp (for time-series queries)
#     """
#     db = get_database()
    
#     # Conversations collection
#     conversations = db["conversations"]
#     await conversations.create_index("conversation_id", unique=True)
#     await conversations.create_index("user_id")
#     await conversations.create_index("created_at")
#     print("✅ Created indexes for 'conversations' collection")
    
#     # Users collection
#     users = db["users"]
#     await users.create_index("user_id", unique=True)
#     await users.create_index("email", unique=True)
#     print("✅ Created indexes for 'users' collection")
    
#     # Retrieval logs collection
#     retrieval_logs = db["retrieval_logs"]
#     await retrieval_logs.create_index("log_id", unique=True)
#     await retrieval_logs.create_index("timestamp")
#     await retrieval_logs.create_index("user_id")
#     print("✅ Created indexes for 'retrieval_logs' collection")
    
#     print("✅ All database indexes created successfully")


# # ============================================================================
# # USAGE EXAMPLES (for reference)
# # ============================================================================
# """
# # In your repository or service:

# from app.db.mongodb import get_database

# async def get_user_conversations(user_id: str):
#     db = get_database()
#     conversations = db["conversations"]
    
#     cursor = conversations.find({"user_id": user_id})
#     results = await cursor.to_list(length=10)
    
#     return results

# # In main.py startup:
# from app.db.mongodb import connect_to_mongo, create_indexes

# await connect_to_mongo()
# await create_indexes()  # Run once on first deployment
# """























# # Key Features:
# # Async/Await - Works with FastAPI's async nature

# # Connection Pooling - Reuses connections efficiently

# # Singleton Pattern - One connection for entire app

# # MongoDB Atlas Compatible - Works with cloud MongoDB

# # Index Creation - Optimizes query performance

# # Health Check - check_connection() for monitoring

# # How to Use:
# # python
# # # In any repository/service file:

# # from app.db.mongodb import get_database

# # async def save_conversation(data: dict):
# #     db = get_database()
# #     collection = db["conversations"]
# #     result = await collection.insert_one(data)
# #     return str(result.inserted_id)

















"""
MongoDB Connection with Motor (Async Driver)
Handles async connection to MongoDB Atlas for conversation storage
"""

import motor.motor_asyncio
from app.config import settings


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
mongodb_client = None
mongodb_database = None


# ============================================================================
# CONNECTION FUNCTIONS
# ============================================================================

async def connect_to_mongo():
    """
    Connect to MongoDB Atlas on application startup.
    
    This is called from main.py during FastAPI lifespan startup.
    Uses Motor for async MongoDB operations.
    
    Returns:
        database: MongoDB database instance or None if connection fails
    """
    global mongodb_client, mongodb_database
    
    try:
        print("\n🔌 Connecting to MongoDB Atlas...")
        
        # Hide password in logs
        # uri_display = settings.MONGODB_URI[:50] + "..." if len(settings.MONGODB_URI) > 50 else settings.MONGODB_URI
        # print(f"   URI: {uri_display}")
        print(f"   Database: {settings.DATABASE_NAME}")
        
        # Create Motor client (async MongoDB driver)
        mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=10000,         # 10 second connection timeout
            socketTimeoutMS=10000           # 10 second socket timeout
        )
        
        # Get database reference
        mongodb_database = mongodb_client[settings.DATABASE_NAME]
        
        # Test connection with ping
        await mongodb_client.admin.command('ping')
        
        print(f"✅ MongoDB connected successfully!")
        print(f"   Database: {settings.DATABASE_NAME}")
        
        return mongodb_database
    
    except Exception as e:
        print(f"\n❌ MongoDB connection FAILED!")
        print(f"   Error: {str(e)}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check MONGODB_URI in .env file")
        print(f"   2. Verify MongoDB Atlas cluster is running")
        print(f"   3. Check network access settings (allow your IP)")
        print(f"   4. Verify database user credentials")
        print(f"\n⚠️  Backend will start but MongoDB features won't work!\n")
        
        # Set to None (app can still start for debugging)
        mongodb_database = None
        return None


async def close_mongo_connection():
    """
    Close MongoDB connection on application shutdown.
    
    This is called from main.py during FastAPI lifespan shutdown.
    """
    global mongodb_client
    
    if mongodb_client:
        print("\n🔌 Closing MongoDB connection...")
        mongodb_client.close()
        print("✅ MongoDB connection closed")
    else:
        print("ℹ️  No MongoDB connection to close")


def get_database():
    """
    Get MongoDB database instance.
    
    This is used by repositories to access the database.
    Returns None if MongoDB is not connected (for graceful degradation).
    
    Returns:
        database: MongoDB database instance or None
    """
    if mongodb_database is None:
        print("\n⚠️  WARNING: MongoDB database not available!")
        print("   Attempting to use database features without connection")
        print("   Make sure MongoDB connection succeeded during startup\n")
    
    return mongodb_database


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In main.py (FastAPI lifespan):

from app.db.mongodb import connect_to_mongo, close_mongo_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()

# In repositories:

from app.db.mongodb import get_database

class SomeRepository:
    def __init__(self):
        self.db = get_database()
        if self.db:
            self.collection = self.db["my_collection"]
"""
