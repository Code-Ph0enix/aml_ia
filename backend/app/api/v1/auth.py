"""
Authentication API Endpoints
User registration, login, and token management
"""

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import timedelta

from app.models.user import UserRegister, UserLogin, Token, UserResponse, TokenData
from app.db.repositories.user_repository import UserRepository
from app.utils.security import verify_password, create_access_token
from app.utils.dependencies import get_current_user
from app.config import settings


router = APIRouter()


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    """
    Register a new user.
    
    Creates a new user account with hashed password and returns
    an access token for immediate login.
    
    Args:
        user_data: User registration data (email, password, full_name)
        
    Returns:
        Token: JWT access token and user info
        
    Raises:
        HTTPException: If email already exists
    """
    user_repo = UserRepository()
    
    try:
        # Create user
        user_id = await user_repo.create_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        # Get created user
        user = await user_repo.get_user_by_id(user_id)
        
        # Generate access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"user_id": user["user_id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        # Return token and user info
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                user_id=user["user_id"],
                email=user["email"],
                full_name=user["full_name"],
                created_at=user["created_at"]
            )
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"❌ Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post("/login", response_model=Token)
async def login_user(user_data: UserLogin):
    """
    Login user and get access token.
    
    Validates user credentials and returns JWT access token.
    
    Args:
        user_data: User login data (email, password)
        
    Returns:
        Token: JWT access token and user info
        
    Raises:
        HTTPException: If credentials are invalid
    """
    user_repo = UserRepository()
    
    # Get user by email
    user = await user_repo.get_user_by_email(user_data.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Generate access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"user_id": user["user_id"], "email": user["email"]},
        expires_delta=access_token_expires
    )
    
    # Return token and user info
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            user_id=user["user_id"],
            email=user["email"],
            full_name=user["full_name"],
            created_at=user["created_at"]
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Protected route that requires valid JWT token.
    
    Args:
        current_user: Current authenticated user (from token)
        
    Returns:
        UserResponse: Current user information
    """
    user_repo = UserRepository()
    user = await user_repo.get_user_by_id(current_user.user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        user_id=user["user_id"],
        email=user["email"],
        full_name=user["full_name"],
        created_at=user["created_at"]
    )


@router.post("/logout")
async def logout_user(current_user: TokenData = Depends(get_current_user)):
    """
    Logout user (client-side token deletion).
    
    In JWT-based auth, logout is handled client-side by
    deleting the token. This endpoint is for logging purposes.
    
    Args:
        current_user: Current authenticated user (from token)
        
    Returns:
        dict: Success message
    """
    print(f"👋 User logged out: {current_user.email}")
    
    return {
        "message": "Successfully logged out",
        "user_id": current_user.user_id
    }
