"""
User Repository - MongoDB CRUD for Users
Handles user registration, retrieval, and management
"""

import uuid
from datetime import datetime
from typing import Optional, Dict
from app.db.mongodb import get_database
from app.utils.security import hash_password


class UserRepository:
    """Repository for user data in MongoDB"""
    
    def __init__(self):
        """Initialize repository with database connection"""
        self.db = get_database()
        
        if self.db is None:
            print("⚠️ UserRepository: MongoDB not connected")
            self.users = None
        else:
            self.users = self.db["users"]
            print("✅ UserRepository initialized")
    
    def _check_connection(self):
        """Check if MongoDB is connected"""
        if self.db is None or self.users is None:
            raise RuntimeError("MongoDB not connected")
    
    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str
    ) -> str:
        """
        Create a new user.
        
        Args:
            email: User email (unique)
            password: Plain text password (will be hashed)
            full_name: User's full name
            
        Returns:
            str: User ID
            
        Raises:
            ValueError: If email already exists
        """
        self._check_connection()
        
        # Check if user already exists
        existing_user = await self.users.find_one({"email": email})
        if existing_user:
            raise ValueError("Email already registered")
        
        # Create user document
        user_id = str(uuid.uuid4())
        user = {
            "user_id": user_id,
            "email": email,
            "hashed_password": hash_password(password),
            "full_name": full_name,
            "created_at": datetime.now(),
            "is_active": True
        }
        
        await self.users.insert_one(user)
        print(f"✅ Created user: {email}")
        
        return user_id
    
    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        Get user by email.
        
        Args:
            email: User email
            
        Returns:
            dict or None: User document
        """
        self._check_connection()
        
        user = await self.users.find_one({"email": email})
        
        if user and "_id" in user:
            user["_id"] = str(user["_id"])
        
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            dict or None: User document
        """
        self._check_connection()
        
        user = await self.users.find_one({"user_id": user_id})
        
        if user and "_id" in user:
            user["_id"] = str(user["_id"])
        
        return user
    
    async def update_user(self, user_id: str, updates: Dict) -> bool:
        """
        Update user information.
        
        Args:
            user_id: User ID
            updates: Dictionary of fields to update
            
        Returns:
            bool: Success status
        """
        self._check_connection()
        
        # Don't allow updating certain fields
        forbidden_fields = ["user_id", "email", "hashed_password", "created_at"]
        for field in forbidden_fields:
            updates.pop(field, None)
        
        result = await self.users.update_one(
            {"user_id": user_id},
            {"$set": updates}
        )
        
        return result.modified_count > 0
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Soft delete a user (mark as inactive).
        
        Args:
            user_id: User ID
            
        Returns:
            bool: Success status
        """
        self._check_connection()
        
        result = await self.users.update_one(
            {"user_id": user_id},
            {"$set": {"is_active": False, "deleted_at": datetime.now()}}
        )
        
        return result.modified_count > 0
