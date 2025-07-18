import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.auth.transport import requests
from google.oauth2 import id_token
import logging

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Google OAuth Configuration
GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")

security = HTTPBearer()

class AuthService:
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.error(f"JWT error: {e}")
            return None

    @staticmethod
    def verify_google_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify Google OAuth token"""
        try:
            # Verify the token with Google
            idinfo = id_token.verify_oauth2_token(
                token, requests.Request(), GOOGLE_CLIENT_ID
            )
            
            # Check if token is from our app
            if idinfo['aud'] != GOOGLE_CLIENT_ID:
                logger.error("Token audience mismatch")
                return None
                
            return {
                "user_id": idinfo['sub'],
                "email": idinfo['email'],
                "name": idinfo.get('name', ''),
                "picture": idinfo.get('picture', ''),
                "email_verified": idinfo.get('email_verified', False)
            }
        except ValueError as e:
            logger.error(f"Google token verification failed: {e}")
            return None

    @staticmethod
    def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """Get current user from JWT token"""
        token = credentials.credentials
        payload = AuthService.verify_token(token)
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload

class SessionManager:
    """Manage user sessions"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, str]:
        """Create a new user session"""
        access_token = AuthService.create_access_token(data={
            "user_id": user_id,
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "session_id": f"session_{user_id}_{datetime.utcnow().timestamp()}"
        })
        
        refresh_token = AuthService.create_refresh_token(data={
            "user_id": user_id,
            "email": user_data.get("email")
        })
        
        # Store session info
        self.active_sessions[user_id] = {
            "user_data": user_data,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    def refresh_session(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access token using refresh token"""
        payload = AuthService.verify_token(refresh_token)
        
        if payload is None or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("user_id")
        if not user_id or user_id not in self.active_sessions:
            return None
        
        # Create new access token
        user_data = self.active_sessions[user_id]["user_data"]
        new_access_token = AuthService.create_access_token(data={
            "user_id": user_id,
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "session_id": f"session_{user_id}_{datetime.utcnow().timestamp()}"
        })
        
        # Update last activity
        self.active_sessions[user_id]["last_activity"] = datetime.utcnow()
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    def revoke_session(self, user_id: str) -> bool:
        """Revoke user session"""
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
            return True
        return False
    
    def is_session_active(self, user_id: str) -> bool:
        """Check if user session is active"""
        return user_id in self.active_sessions

# Global session manager instance
session_manager = SessionManager()

# Dependency for protected routes
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    return AuthService.get_current_user(credentials)

# Optional dependency for protected routes
def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Optional dependency to get current authenticated user"""
    if credentials is None:
        return None
    
    try:
        return AuthService.get_current_user(credentials)
    except HTTPException:
        return None
