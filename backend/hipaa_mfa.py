"""
HIPAA-Compliant Multi-Factor Authentication Module
Implements TOTP-based MFA with backup codes for medical applications
"""

import os
import secrets
import pyotp
import qrcode
import io
import base64
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from cryptography.fernet import Fernet
import logging
from hipaa_security import HIPAAAuditLogger, HIPAAEncryption

logger = logging.getLogger(__name__)

class HIPAAMFAManager:
    """HIPAA-compliant Multi-Factor Authentication manager"""
    
    def __init__(self, db_path: str = "data/mfa.db"):
        self.db_path = db_path
        self.encryption = HIPAAEncryption()
        self.audit_logger = HIPAAAuditLogger()
        self.issuer_name = os.getenv("MFA_ISSUER_NAME", "Medical_Chatbot_HIPAA")
        self.backup_codes_count = int(os.getenv("MFA_BACKUP_CODES_COUNT", "10"))
        self._init_database()
    
    def _init_database(self):
        """Initialize MFA database with proper schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_mfa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                secret_key TEXT NOT NULL,
                backup_codes TEXT,
                is_enabled BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME,
                failed_attempts INTEGER DEFAULT 0
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS mfa_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                ip_address TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("MFA database initialized")
    
    def setup_mfa_for_user(self, user_id: str) -> Dict:
        """Set up MFA for a user and return QR code and backup codes"""
        
        # Generate secret key for TOTP
        secret_key = pyotp.random_base32()
        
        # Generate backup codes
        backup_codes = self._generate_backup_codes()
        
        # Encrypt sensitive data
        encrypted_secret = self.encryption.encrypt_data(secret_key)
        encrypted_backup_codes = self.encryption.encrypt_data(json.dumps(backup_codes))
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO user_mfa 
                (user_id, secret_key, backup_codes, is_enabled, created_at, failed_attempts)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, encrypted_secret, encrypted_backup_codes, False, datetime.utcnow(), 0))
            conn.commit()
        finally:
            conn.close()
        
        # Generate QR code
        totp = pyotp.TOTP(secret_key)
        provisioning_uri = totp.provisioning_uri(
            name=user_id,
            issuer_name=self.issuer_name
        )
        
        qr_code_data = self._generate_qr_code(provisioning_uri)
        
        # Log MFA setup
        self.audit_logger.log_phi_access(
            user_id, 'mfa_setup_initiated', 'user_authentication',
            details={'issuer': self.issuer_name, 'backup_codes_generated': len(backup_codes)}
        )
        
        return {
            'secret_key': secret_key,  # Show once for manual entry
            'qr_code': qr_code_data,
            'backup_codes': backup_codes,
            'setup_complete': False
        }
    
    def verify_and_enable_mfa(self, user_id: str, token: str) -> bool:
        """Verify setup token and enable MFA for user"""
        
        # Get user's MFA data
        mfa_data = self._get_user_mfa_data(user_id)
        if not mfa_data:
            self._log_mfa_attempt(user_id, 'setup_verification', False, 'No MFA data found')
            return False
        
        secret_key = self.encryption.decrypt_data(mfa_data['secret_key'])
        
        # Verify the token
        totp = pyotp.TOTP(secret_key)
        
        if totp.verify(token, valid_window=1):  # Allow 30-second window
            # Enable MFA for the user
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    UPDATE user_mfa 
                    SET is_enabled = ?, last_used = ?, failed_attempts = 0
                    WHERE user_id = ?
                ''', (True, datetime.utcnow(), user_id))
                conn.commit()
            finally:
                conn.close()
            
            self._log_mfa_attempt(user_id, 'setup_verification', True, 'MFA enabled successfully')
            self.audit_logger.log_phi_access(
                user_id, 'mfa_enabled', 'user_authentication',
                details={'enabled_at': datetime.utcnow().isoformat()}
            )
            return True
        else:
            self._increment_failed_attempts(user_id)
            self._log_mfa_attempt(user_id, 'setup_verification', False, 'Invalid token')
            return False
    
    def verify_mfa_token(self, user_id: str, token: str, is_backup_code: bool = False) -> bool:
        """Verify MFA token or backup code"""
        
        mfa_data = self._get_user_mfa_data(user_id)
        if not mfa_data or not mfa_data['is_enabled']:
            self._log_mfa_attempt(user_id, 'verification', False, 'MFA not enabled')
            return False
        
        # Check if user is locked out
        if mfa_data['failed_attempts'] >= 5:  # HIPAA security: lock after 5 failed attempts
            self._log_mfa_attempt(user_id, 'verification', False, 'Account locked due to failed attempts')
            return False
        
        if is_backup_code:
            return self._verify_backup_code(user_id, token, mfa_data)
        else:
            return self._verify_totp_token(user_id, token, mfa_data)
    
    def _verify_totp_token(self, user_id: str, token: str, mfa_data: Dict) -> bool:
        """Verify TOTP token"""
        secret_key = self.encryption.decrypt_data(mfa_data['secret_key'])
        totp = pyotp.TOTP(secret_key)
        
        if totp.verify(token, valid_window=1):
            # Reset failed attempts and update last used
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    UPDATE user_mfa 
                    SET last_used = ?, failed_attempts = 0
                    WHERE user_id = ?
                ''', (datetime.utcnow(), user_id))
                conn.commit()
            finally:
                conn.close()
            
            self._log_mfa_attempt(user_id, 'totp_verification', True, 'Token verified successfully')
            return True
        else:
            self._increment_failed_attempts(user_id)
            self._log_mfa_attempt(user_id, 'totp_verification', False, 'Invalid TOTP token')
            return False
    
    def _verify_backup_code(self, user_id: str, code: str, mfa_data: Dict) -> bool:
        """Verify and consume backup code"""
        encrypted_backup_codes = mfa_data['backup_codes']
        backup_codes_json = self.encryption.decrypt_data(encrypted_backup_codes)
        backup_codes = json.loads(backup_codes_json)
        
        # Hash the provided code for comparison
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        if code_hash in backup_codes:
            # Remove used backup code
            backup_codes.remove(code_hash)
            
            # Update database
            updated_codes = self.encryption.encrypt_data(json.dumps(backup_codes))
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    UPDATE user_mfa 
                    SET backup_codes = ?, last_used = ?, failed_attempts = 0
                    WHERE user_id = ?
                ''', (updated_codes, datetime.utcnow(), user_id))
                conn.commit()
            finally:
                conn.close()
            
            self._log_mfa_attempt(user_id, 'backup_code_verification', True, 
                                f'Backup code used, {len(backup_codes)} remaining')
            
            # Log warning if running low on backup codes
            if len(backup_codes) <= 2:
                self.audit_logger.log_phi_access(
                    user_id, 'mfa_backup_codes_low', 'user_authentication',
                    details={'remaining_codes': len(backup_codes)}
                )
            
            return True
        else:
            self._increment_failed_attempts(user_id)
            self._log_mfa_attempt(user_id, 'backup_code_verification', False, 'Invalid backup code')
            return False
    
    def regenerate_backup_codes(self, user_id: str) -> List[str]:
        """Regenerate backup codes for user"""
        
        mfa_data = self._get_user_mfa_data(user_id)
        if not mfa_data:
            return []
        
        # Generate new backup codes
        backup_codes = self._generate_backup_codes()
        encrypted_backup_codes = self.encryption.encrypt_data(json.dumps(backup_codes))
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                UPDATE user_mfa 
                SET backup_codes = ?
                WHERE user_id = ?
            ''', (encrypted_backup_codes, user_id))
            conn.commit()
        finally:
            conn.close()
        
        # Log regeneration
        self.audit_logger.log_phi_access(
            user_id, 'mfa_backup_codes_regenerated', 'user_authentication',
            details={'new_codes_count': len(backup_codes)}
        )
        
        # Return readable codes (not hashed)
        return [secrets.token_hex(4).upper() for _ in range(self.backup_codes_count)]
    
    def disable_mfa(self, user_id: str, admin_override: bool = False) -> bool:
        """Disable MFA for user (requires admin override for security)"""
        
        if not admin_override:
            logger.warning(f"MFA disable attempted without admin override for user: {user_id}")
            return False
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('DELETE FROM user_mfa WHERE user_id = ?', (user_id,))
            conn.commit()
            deleted = conn.total_changes > 0
        finally:
            conn.close()
        
        if deleted:
            self.audit_logger.log_phi_access(
                user_id, 'mfa_disabled', 'user_authentication',
                details={'admin_override': admin_override}
            )
            logger.warning(f"MFA disabled for user: {user_id}")
        
        return deleted
    
    def get_mfa_status(self, user_id: str) -> Dict:
        """Get MFA status for user"""
        mfa_data = self._get_user_mfa_data(user_id)
        
        if not mfa_data:
            return {'enabled': False, 'setup_required': True}
        
        # Count remaining backup codes
        backup_codes_json = self.encryption.decrypt_data(mfa_data['backup_codes'])
        backup_codes = json.loads(backup_codes_json)
        
        return {
            'enabled': mfa_data['is_enabled'],
            'setup_required': not mfa_data['is_enabled'],
            'last_used': mfa_data['last_used'],
            'failed_attempts': mfa_data['failed_attempts'],
            'locked': mfa_data['failed_attempts'] >= 5,
            'backup_codes_remaining': len(backup_codes),
            'backup_codes_low': len(backup_codes) <= 2
        }
    
    def unlock_user_mfa(self, user_id: str, admin_user_id: str) -> bool:
        """Unlock user MFA after failed attempts (admin only)"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                UPDATE user_mfa 
                SET failed_attempts = 0
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
            unlocked = conn.total_changes > 0
        finally:
            conn.close()
        
        if unlocked:
            self.audit_logger.log_phi_access(
                admin_user_id, 'mfa_user_unlocked', 'user_authentication',
                details={'unlocked_user': user_id}
            )
            logger.info(f"MFA unlocked for user: {user_id} by admin: {admin_user_id}")
        
        return unlocked
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate secure backup codes"""
        codes = []
        for _ in range(self.backup_codes_count):
            # Generate 8-digit backup code
            code = secrets.token_hex(4).upper()
            # Store hash for security
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            codes.append(code_hash)
        return codes
    
    def _generate_qr_code(self, provisioning_uri: str) -> str:
        """Generate QR code for TOTP setup"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for web display
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _get_user_mfa_data(self, user_id: str) -> Optional[Dict]:
        """Get user's MFA data from database"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT secret_key, backup_codes, is_enabled, last_used, failed_attempts
                FROM user_mfa WHERE user_id = ?
            ''', (user_id,))
            result = cursor.fetchone()
            
            if result:
                return {
                    'secret_key': result[0],
                    'backup_codes': result[1],
                    'is_enabled': bool(result[2]),
                    'last_used': result[3],
                    'failed_attempts': result[4]
                }
            return None
        finally:
            conn.close()
    
    def _increment_failed_attempts(self, user_id: str):
        """Increment failed MFA attempts"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                UPDATE user_mfa 
                SET failed_attempts = failed_attempts + 1
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
        finally:
            conn.close()
    
    def _log_mfa_attempt(self, user_id: str, attempt_type: str, success: bool, details: str):
        """Log MFA attempt"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT INTO mfa_attempts 
                (user_id, attempt_type, success, ip_address, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, attempt_type, success, 'localhost', details))
            conn.commit()
        finally:
            conn.close()

# Global MFA manager instance
mfa_manager = HIPAAMFAManager()

def require_mfa_verification(user_id: str, token: str, is_backup_code: bool = False) -> bool:
    """Decorator function to require MFA verification"""
    return mfa_manager.verify_mfa_token(user_id, token, is_backup_code)

if __name__ == "__main__":
    # Test MFA functionality
    test_user = "test_doctor_123"
    
    print("Testing HIPAA MFA System...")
    
    # Setup MFA
    setup_result = mfa_manager.setup_mfa_for_user(test_user)
    print(f"MFA Setup initiated for user: {test_user}")
    print(f"Secret key: {setup_result['secret_key']}")
    print(f"Backup codes generated: {len(setup_result['backup_codes'])}")
    
    # Test verification (this would normally be done with a real TOTP app)
    import pyotp
    totp = pyotp.TOTP(setup_result['secret_key'])
    test_token = totp.now()
    
    print(f"Generated test token: {test_token}")
    
    # Verify and enable
    enabled = mfa_manager.verify_and_enable_mfa(test_user, test_token)
    print(f"MFA enabled: {enabled}")
    
    # Check status
    status = mfa_manager.get_mfa_status(test_user)
    print(f"MFA Status: {status}")