"""
FastAPI Dependencies
Authentication and authorization dependencies
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from app.utils.security import decode_access_token
from app.db.repositories.user_repository import UserRepository
from app.models.user import TokenData


# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """
    Get current authenticated user from JWT token.
    
    This dependency extracts and validates the JWT token from the
    Authorization header and returns the user data.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        TokenData: User data from token
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode token
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        raise credentials_exception
    
    # Extract user data
    user_id: str = payload.get("user_id")
    email: str = payload.get("email")
    
    if user_id is None or email is None:
        raise credentials_exception
    
    # Verify user exists
    user_repo = UserRepository()
    user = await user_repo.get_user_by_id(user_id)
    
    if user is None or not user.get("is_active", False):
        raise credentials_exception
    
    return TokenData(user_id=user_id, email=email)


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """
    Get current user if authenticated, None otherwise.
    
    This is a non-required version of get_current_user for optional auth.
    
    Args:
        credentials: Optional HTTP Bearer credentials
        
    Returns:
        TokenData or None: User data from token or None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
