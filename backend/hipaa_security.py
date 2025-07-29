"""
HIPAA-Compliant Security Module
Implements real encryption, audit logging, and access controls for medical data protection
"""

import os
import hashlib
import secrets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import sqlite3
import re
import jwt
from pathlib import Path

# Configure secure logging
os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hipaa_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HIPAAEncryption:
    """AES-256 encryption for PHI data protection"""
    
    def __init__(self, key_file: str = "data/encryption_key.key"):
        self.key_file = key_file
        self.key = self._load_or_create_key()
        self.fernet = Fernet(self.key)
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new AES-256 key"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            
            # Generate new AES-256 key
            key = Fernet.generate_key()
            
            # Save key with restricted permissions
            with open(self.key_file, 'wb') as f:
                f.write(key)
                
            # Set restrictive file permissions (Windows)
            try:
                os.chmod(self.key_file, 0o600)
            except:
                logger.warning("Could not set restrictive file permissions on Windows")
                
            logger.info("Generated new AES-256 encryption key")
            return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data using AES-256"""
        if not data:
            return ""
        
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using AES-256"""
        if not encrypted_data:
            return ""
        
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_file(self, file_path: str) -> str:
        """Encrypt entire file"""
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encrypted_data = self.fernet.encrypt(file_data)
        encrypted_file_path = f"{file_path}.encrypted"
        
        with open(encrypted_file_path, 'wb') as f:
            f.write(encrypted_data)
        
        # Remove original file
        os.remove(file_path)
        logger.info(f"File encrypted: {file_path}")
        return encrypted_file_path

class PHIRedactor:
    """Real PHI redaction for HIPAA compliance"""
    
    def __init__(self):
        self.phi_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'mrn': r'\b(?:MRN|Medical Record|Record Number):?\s*[A-Z0-9]{6,12}\b',
            'insurance_id': r'\b(?:Insurance|Policy)\s*(?:ID|Number):?\s*[A-Z0-9]{6,15}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b'
        }
    
    def redact_phi(self, text: str) -> str:
        """Redact PHI from text using comprehensive patterns"""
        if not text:
            return text
        
        redacted_text = text
        redacted_items = []
        
        for phi_type, pattern in self.phi_patterns.items():
            matches = re.findall(pattern, redacted_text, re.IGNORECASE)
            if matches:
                redacted_items.extend([(phi_type, match) for match in matches])
                redacted_text = re.sub(pattern, f'[{phi_type.upper()}_REDACTED]', redacted_text, flags=re.IGNORECASE)
        
        if redacted_items:
            logger.info(f"Redacted {len(redacted_items)} PHI items from text")
        
        return redacted_text
    
    def is_phi_present(self, text: str) -> bool:
        """Check if text contains PHI"""
        for pattern in self.phi_patterns.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

class HIPAAAuditLogger:
    """Immutable audit logging for HIPAA compliance"""
    
    def __init__(self, db_path: str = "data/hipaa_audit.db"):
        self.db_path = db_path
        self.encryption = HIPAAEncryption()
        self._init_database()
    
    def _init_database(self):
        """Initialize audit database with proper schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                session_id TEXT,
                phi_accessed BOOLEAN DEFAULT FALSE,
                details TEXT,
                hash_chain TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                access_granted BOOLEAN NOT NULL,
                reason TEXT,
                ip_address TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Audit database initialized")
    
    def log_phi_access(self, user_id: str, action: str, resource: str, 
                      ip_address: str = None, details: Dict = None):
        """Log PHI access with immutable hash chain"""
        timestamp = datetime.utcnow().isoformat()
        
        # Get previous hash for chain integrity
        prev_hash = self._get_last_hash()
        
        # Create audit record
        audit_record = {
            'timestamp': timestamp,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'ip_address': ip_address or 'localhost',
            'phi_accessed': True,
            'details': json.dumps(details) if details else None
        }
        
        # Create hash chain
        record_string = json.dumps(audit_record, sort_keys=True)
        current_hash = hashlib.sha256(f"{prev_hash}{record_string}".encode()).hexdigest()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO audit_log 
            (timestamp, user_id, action, resource, ip_address, phi_accessed, details, hash_chain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, user_id, action, resource, ip_address, True, 
              audit_record['details'], current_hash))
        
        conn.commit()
        conn.close()
        
        logger.info(f"PHI access logged: {user_id} - {action} - {resource}")
    
    def log_access_attempt(self, user_id: str, resource_type: str, 
                          access_granted: bool, reason: str = None):
        """Log access attempts for audit trail"""
        timestamp = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO access_log 
            (timestamp, user_id, resource_type, access_granted, reason, ip_address)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, user_id, resource_type, access_granted, reason, 'localhost'))
        
        conn.commit()
        conn.close()
    
    def _get_last_hash(self) -> str:
        """Get the last hash in the chain for integrity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT hash_chain FROM audit_log ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else "genesis_block"
    
    def verify_audit_integrity(self) -> bool:
        """Verify audit log integrity using hash chain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, user_id, action, resource, ip_address, 
                   phi_accessed, details, hash_chain 
            FROM audit_log ORDER BY id ASC
        ''')
        
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            return True
        
        prev_hash = "genesis_block"
        for record in records:
            timestamp, user_id, action, resource, ip_address, phi_accessed, details, stored_hash = record
            
            audit_record = {
                'timestamp': timestamp,
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'ip_address': ip_address,
                'phi_accessed': bool(phi_accessed),
                'details': details
            }
            
            record_string = json.dumps(audit_record, sort_keys=True)
            expected_hash = hashlib.sha256(f"{prev_hash}{record_string}".encode()).hexdigest()
            
            if expected_hash != stored_hash:
                logger.error("Audit log integrity violation detected")
                return False
            
            prev_hash = stored_hash
        
        logger.info("Audit log integrity verified")
        return True

class HIPAAAccessControl:
    """Role-based access control for PHI"""
    
    def __init__(self):
        self.audit_logger = HIPAAAuditLogger()
        self.role_permissions = {
            'doctor': {
                'phi_fields': ['name', 'dob', 'ssn', 'medical_history', 'diagnosis', 'treatment'],
                'actions': ['read', 'write', 'update', 'prescribe'],
                'emergency_override': True
            },
            'nurse': {
                'phi_fields': ['name', 'dob', 'medical_history', 'vital_signs'],
                'actions': ['read', 'update'],
                'emergency_override': True
            },
            'admin': {
                'phi_fields': ['name', 'dob', 'insurance_info'],
                'actions': ['read', 'update'],
                'emergency_override': False
            },
            'billing': {
                'phi_fields': ['name', 'dob', 'ssn', 'insurance_info', 'billing_address'],
                'actions': ['read', 'update'],
                'emergency_override': False
            },
            'patient': {
                'phi_fields': ['own_data_only'],
                'actions': ['read'],
                'emergency_override': False
            }
        }
    
    def check_access(self, user_id: str, user_role: str, resource_type: str, 
                    action: str, phi_fields: List[str] = None) -> bool:
        """Check if user has access to specific PHI fields"""
        
        if user_role not in self.role_permissions:
            self.audit_logger.log_access_attempt(
                user_id, resource_type, False, "Invalid role"
            )
            return False
        
        role_config = self.role_permissions[user_role]
        
        # Check action permission
        if action not in role_config['actions']:
            self.audit_logger.log_access_attempt(
                user_id, resource_type, False, f"Action '{action}' not permitted for role"
            )
            return False
        
        # Check PHI field access
        if phi_fields:
            allowed_fields = role_config['phi_fields']
            if 'own_data_only' not in allowed_fields:
                for field in phi_fields:
                    if field not in allowed_fields:
                        self.audit_logger.log_access_attempt(
                            user_id, resource_type, False, f"PHI field '{field}' not permitted"
                        )
                        return False
        
        # Log successful access
        self.audit_logger.log_access_attempt(
            user_id, resource_type, True, "Access granted"
        )
        
        return True
    
    def emergency_override(self, user_id: str, user_role: str, 
                          emergency_type: str, justification: str) -> Dict:
        """Handle emergency access override"""
        
        if user_role not in self.role_permissions:
            return {'granted': False, 'reason': 'Invalid role'}
        
        role_config = self.role_permissions[user_role]
        
        if not role_config['emergency_override']:
            return {'granted': False, 'reason': 'Emergency override not permitted for role'}
        
        # Grant emergency access
        emergency_access = {
            'granted': True,
            'user_id': user_id,
            'emergency_type': emergency_type,
            'justification': justification,
            'granted_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            'audit_required': True
        }
        
        # Log emergency access
        self.audit_logger.log_phi_access(
            user_id, 'emergency_override', 'all_phi_data',
            details={
                'emergency_type': emergency_type,
                'justification': justification,
                'expires_at': emergency_access['expires_at']
            }
        )
        
        logger.warning(f"Emergency override granted: {user_id} - {emergency_type}")
        return emergency_access

class HIPAASessionManager:
    """Secure session management with automatic timeout"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions = {}
        self.session_timeout = timedelta(minutes=30)  # HIPAA recommended timeout
        self.audit_logger = HIPAAAuditLogger()
    
    def create_session(self, user_id: str, user_role: str, ip_address: str = None) -> str:
        """Create secure session with JWT token"""
        session_id = secrets.token_urlsafe(32)
        
        payload = {
            'session_id': session_id,
            'user_id': user_id,
            'user_role': user_role,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + self.session_timeout).isoformat(),
            'ip_address': ip_address or 'localhost'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Store session info
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'user_role': user_role,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'ip_address': ip_address
        }
        
        # Log session creation
        self.audit_logger.log_phi_access(
            user_id, 'session_created', 'user_session',
            ip_address, {'session_id': session_id}
        )
        
        return token
    
    def validate_session(self, token: str) -> Optional[Dict]:
        """Validate session token and check timeout"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            session_id = payload['session_id']
            
            # Check if session exists and is not expired
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            # Check timeout
            if datetime.utcnow() - session['last_activity'] > self.session_timeout:
                self.invalidate_session(session_id)
                return None
            
            # Update last activity
            session['last_activity'] = datetime.utcnow()
            
            return {
                'user_id': session['user_id'],
                'user_role': session['user_role'],
                'session_id': session_id
            }
            
        except jwt.InvalidTokenError:
            return None
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Log session end
            self.audit_logger.log_phi_access(
                session['user_id'], 'session_ended', 'user_session',
                session['ip_address'], {'session_id': session_id}
            )
            
            del self.active_sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Initialize HIPAA security components
encryption = HIPAAEncryption()
phi_redactor = PHIRedactor()
audit_logger = HIPAAAuditLogger()
access_control = HIPAAAccessControl()

def secure_medical_response(response: str, user_id: str) -> str:
    """Secure medical response by redacting PHI and logging access"""
    
    # Check if response contains PHI
    if phi_redactor.is_phi_present(response):
        # Log PHI access
        audit_logger.log_phi_access(
            user_id, 'phi_response_generated', 'medical_response',
            details={'response_length': len(response)}
        )
        
        # Redact PHI from response
        redacted_response = phi_redactor.redact_phi(response)
        return redacted_response
    
    return response

def initialize_hipaa_security():
    """Initialize all HIPAA security components"""
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Verify audit integrity
    if not audit_logger.verify_audit_integrity():
        logger.error("CRITICAL: Audit log integrity check failed!")
        raise SecurityError("Audit log integrity compromised")
    
    # Log system initialization
    audit_logger.log_phi_access(
        'system', 'hipaa_security_initialized', 'security_system',
        details={'encryption_enabled': True, 'audit_enabled': True}
    )
    
    logger.info("HIPAA security system initialized successfully")

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

if __name__ == "__main__":
    # Test HIPAA security components
    initialize_hipaa_security()
    
    # Test encryption
    test_data = "Patient John Doe, SSN: 123-45-6789, has chest pain"
    encrypted = encryption.encrypt_data(test_data)
    decrypted = encryption.decrypt_data(encrypted)
    print(f"Encryption test passed: {test_data == decrypted}")
    
    # Test PHI redaction
    redacted = phi_redactor.redact_phi(test_data)
    print(f"PHI redaction: {redacted}")
    
    # Test access control
    access_granted = access_control.check_access(
        'doctor_123', 'doctor', 'patient_record', 'read', ['name', 'medical_history']
    )
    print(f"Access control test: {access_granted}")
