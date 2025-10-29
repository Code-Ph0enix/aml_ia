# """
# Security Utilities
# Password hashing and JWT token management
# """

# from datetime import datetime, timedelta
# from typing import Optional
# from jose import JWTError, jwt
# from passlib.context import CryptContext

# from app.config import settings


# # Password hashing context
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# # ============================================================================
# # PASSWORD HASHING
# # ============================================================================

# def hash_password(password: str) -> str:
#     """
#     Hash a password using bcrypt.
    
#     Args:
#         password: Plain text password
        
#     Returns:
#         str: Hashed password
#     """
#     return pwd_context.hash(password)


# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """
#     Verify a password against a hash.
    
#     Args:
#         plain_password: Plain text password to verify
#         hashed_password: Hashed password from database
        
#     Returns:
#         bool: True if password matches, False otherwise
#     """
#     return pwd_context.verify(plain_password, hashed_password)


# # ============================================================================
# # JWT TOKEN MANAGEMENT
# # ============================================================================

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     """
#     Create a JWT access token.
    
#     Args:
#         data: Data to encode in token (user_id, email, etc.)
#         expires_delta: Optional custom expiration time
        
#     Returns:
#         str: Encoded JWT token
#     """
#     to_encode = data.copy()
    
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
#     return encoded_jwt


# def decode_access_token(token: str) -> Optional[dict]:
#     """
#     Decode and verify a JWT token.
    
#     Args:
#         token: JWT token to decode
        
#     Returns:
#         dict: Decoded token data or None if invalid
#     """
#     try:
#         payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
#         return payload
#     except JWTError:
#         return None









"""
Security utilities for password hashing and JWT tokens
"""

from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from app.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password
    """
    # Bcrypt has a 72 byte limit, truncate if longer
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
    
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored hashed password
    
    Returns:
        True if password matches, False otherwise
    """
    # Truncate to 72 bytes for bcrypt
    if len(plain_password.encode('utf-8')) > 72:
        plain_password = plain_password[:72]
    
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in the token (usually user_id, email)
        expires_delta: Token expiration time (default: from settings)
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token data (dict)
    
    Raises:
        JWTError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None
