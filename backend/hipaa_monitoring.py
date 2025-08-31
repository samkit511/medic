"""
HIPAA-Compliant Log Monitoring and Alerting System
Real-time monitoring of security events and suspicious activities
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable

from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import time
import re
from hipaa_security import HIPAAAuditLogger

# Configure monitoring logger
monitoring_logger = logging.getLogger('hipaa_monitoring')
monitoring_logger.setLevel(logging.INFO)

# Optional email imports
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    monitoring_logger.warning("Email functionality not available")

@dataclass
class SecurityAlert:
    """Security alert data structure"""
    alert_id: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    user_id: str
    description: str
    timestamp: datetime
    details: Dict
    resolved: bool = False

@dataclass
class MonitoringRule:
    """Monitoring rule configuration"""
    rule_id: str
    rule_name: str
    description: str
    pattern: str
    threshold: int
    time_window_minutes: int
    severity: str
    enabled: bool = True

class HIPAASecurityMonitor:
    """Real-time security monitoring and alerting system"""
    
    def __init__(self, db_path: str = "data/security_monitoring.db"):
        self.db_path = db_path
        self.audit_logger = HIPAAAuditLogger()
        self.active_alerts = {}
        self.monitoring_rules = {}
        self.event_counters = defaultdict(lambda: deque())
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.alert_recipients = os.getenv("ALERT_RECIPIENTS", "").split(",")
        
        self._init_database()
        self._load_default_rules()
        
    def _init_database(self):
        """Initialize monitoring database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Security alerts table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS security_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                user_id TEXT NOT NULL,
                description TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                details TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at DATETIME,
                resolved_by TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Monitoring rules table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT UNIQUE NOT NULL,
                rule_name TEXT NOT NULL,
                description TEXT NOT NULL,
                pattern TEXT NOT NULL,
                threshold INTEGER NOT NULL,
                time_window_minutes INTEGER NOT NULL,
                severity TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Event statistics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS event_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                user_id TEXT,
                count INTEGER NOT NULL,
                time_period TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        monitoring_logger.info("Security monitoring database initialized")
    
    def _load_default_rules(self):
        """Load default HIPAA monitoring rules"""
        default_rules = [
            MonitoringRule(
                rule_id="failed_login_attempts",
                rule_name="Failed Login Attempts",
                description="Multiple failed login attempts from same user",
                pattern="login_failed",
                threshold=3,
                time_window_minutes=15,
                severity="HIGH"
            ),
            MonitoringRule(
                rule_id="phi_access_anomaly",
                rule_name="PHI Access Anomaly",
                description="Unusual PHI access patterns",
                pattern="phi_accessed",
                threshold=50,
                time_window_minutes=60,
                severity="MEDIUM"
            ),
            MonitoringRule(
                rule_id="unauthorized_access_attempt",
                rule_name="Unauthorized Access Attempt",
                description="Access attempts to restricted resources",
                pattern="access_denied",
                threshold=5,
                time_window_minutes=10,
                severity="HIGH"
            ),
            MonitoringRule(
                rule_id="mfa_bypass_attempt",
                rule_name="MFA Bypass Attempt",
                description="Multiple MFA verification failures",
                pattern="mfa_verification_failed",
                threshold=5,
                time_window_minutes=30,
                severity="CRITICAL"
            ),
            MonitoringRule(
                rule_id="admin_privilege_escalation",
                rule_name="Admin Privilege Escalation",
                description="Attempts to escalate privileges",
                pattern="privilege_escalation_attempt",
                threshold=1,
                time_window_minutes=5,
                severity="CRITICAL"
            ),
            MonitoringRule(
                rule_id="data_export_anomaly",
                rule_name="Large Data Export",
                description="Unusual data export activity",
                pattern="data_export",
                threshold=10,
                time_window_minutes=60,
                severity="MEDIUM"
            ),
            MonitoringRule(
                rule_id="session_hijacking",
                rule_name="Session Hijacking Attempt",
                description="Suspicious session activity",
                pattern="session_anomaly",
                threshold=3,
                time_window_minutes=15,
                severity="HIGH"
            ),
            MonitoringRule(
                rule_id="off_hours_access",
                rule_name="Off-Hours Access",
                description="PHI access during off-hours",
                pattern="off_hours_phi_access",
                threshold=5,
                time_window_minutes=60,
                severity="MEDIUM"
            )
        ]
        
        # Store rules in database and memory
        conn = sqlite3.connect(self.db_path)
        for rule in default_rules:
            try:
                conn.execute('''
                    INSERT OR IGNORE INTO monitoring_rules 
                    (rule_id, rule_name, description, pattern, threshold, time_window_minutes, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (rule.rule_id, rule.rule_name, rule.description, rule.pattern,
                      rule.threshold, rule.time_window_minutes, rule.severity))
                
                self.monitoring_rules[rule.rule_id] = rule
                
            except Exception as e:
                monitoring_logger.error(f"Failed to load rule {rule.rule_id}: {e}")
        
        conn.commit()
        conn.close()
        monitoring_logger.info(f"Loaded {len(default_rules)} monitoring rules")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring_active:
            monitoring_logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        monitoring_logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        monitoring_logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_audit_logs()
                self._analyze_patterns()
                self._cleanup_old_events()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                monitoring_logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_audit_logs(self):
        """Check recent audit log entries"""
        try:
            # Get recent audit entries
            conn = sqlite3.connect("data/hipaa_audit.db")
            cursor = conn.cursor()
            
            # Check last 5 minutes of audit logs
            five_minutes_ago = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
            
            cursor.execute('''
                SELECT user_id, action, resource, details, timestamp
                FROM audit_log 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (five_minutes_ago,))
            
            recent_entries = cursor.fetchall()
            conn.close()
            
            # Process each entry
            for entry in recent_entries:
                user_id, action, resource, details, timestamp = entry
                self._process_audit_event(user_id, action, resource, details, timestamp)
                
        except Exception as e:
            monitoring_logger.error(f"Error checking audit logs: {e}")
    
    def _process_audit_event(self, user_id: str, action: str, resource: str, 
                           details: str, timestamp: str):
        """Process individual audit event"""
        event_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00').replace('+00:00', ''))
        
        # Classify event type
        event_type = self._classify_event(action, resource, details)
        
        if event_type:
            # Add to event counters
            self.event_counters[f"{event_type}_{user_id}"].append(event_time)
            
            # Check for off-hours access
            if self._is_off_hours(event_time) and "phi" in resource.lower():
                self.event_counters[f"off_hours_phi_access_{user_id}"].append(event_time)
            
            # Check monitoring rules
            self._check_rules_for_event(event_type, user_id, event_time, details)
    
    def _classify_event(self, action: str, resource: str, details: str) -> Optional[str]:
        """Classify audit event type"""
        action_lower = action.lower()
        resource_lower = resource.lower()
        
        # Classification rules
        if "login" in action_lower and "failed" in action_lower:
            return "login_failed"
        elif "access" in action_lower and "denied" in action_lower:
            return "access_denied"
        elif "phi" in resource_lower and "access" in action_lower:
            return "phi_accessed"
        elif "mfa" in action_lower and "failed" in action_lower:
            return "mfa_verification_failed"
        elif "privilege" in action_lower or "escalation" in action_lower:
            return "privilege_escalation_attempt"
        elif "export" in action_lower or "download" in action_lower:
            return "data_export"
        elif "session" in resource_lower and "anomaly" in action_lower:
            return "session_anomaly"
        
        return None
    
    def _is_off_hours(self, event_time: datetime) -> bool:
        """Check if event occurred during off-hours"""
        hour = event_time.hour
        weekday = event_time.weekday()
        
        # Define business hours: 8 AM to 6 PM, Monday to Friday
        if weekday >= 5:  # Weekend
            return True
        if hour < 8 or hour >= 18:  # Outside business hours
            return True
        
        return False
    
    def _check_rules_for_event(self, event_type: str, user_id: str, 
                              event_time: datetime, details: str):
        """Check monitoring rules against event"""
        for rule in self.monitoring_rules.values():
            if not rule.enabled:
                continue
            
            if rule.pattern in event_type:
                self._evaluate_rule(rule, event_type, user_id, event_time, details)
    
    def _evaluate_rule(self, rule: MonitoringRule, event_type: str, 
                      user_id: str, event_time: datetime, details: str):
        """Evaluate monitoring rule"""
        key = f"{event_type}_{user_id}"
        events = self.event_counters[key]
        
        # Count events within time window
        time_threshold = event_time - timedelta(minutes=rule.time_window_minutes)
        recent_events = [e for e in events if e >= time_threshold]
        
        if len(recent_events) >= rule.threshold:
            self._trigger_alert(rule, user_id, len(recent_events), details)
    
    def _trigger_alert(self, rule: MonitoringRule, user_id: str, 
                      event_count: int, details: str):
        """Trigger security alert"""
        alert_id = f"{rule.rule_id}_{user_id}_{int(time.time())}"
        
        # Check if similar alert already exists
        if self._has_active_alert(rule.rule_id, user_id):
            return
        
        alert = SecurityAlert(
            alert_id=alert_id,
            alert_type=rule.rule_id,
            severity=rule.severity,
            user_id=user_id,
            description=f"{rule.description} - {event_count} events in {rule.time_window_minutes} minutes",
            timestamp=datetime.utcnow(),
            details={
                'rule_name': rule.rule_name,
                'event_count': event_count,
                'threshold': rule.threshold,
                'time_window': rule.time_window_minutes,
                'raw_details': details
            }
        )
        
        # Store alert
        self._store_alert(alert)
        
        # Send notifications
        self._send_alert_notification(alert)
        
        # Log to audit system
        self.audit_logger.log_phi_access(
            'security_monitor', 'security_alert_triggered', 'monitoring_system',
            details={
                'alert_id': alert_id,
                'alert_type': rule.rule_id,
                'severity': rule.severity,
                'affected_user': user_id
            }
        )
        
        monitoring_logger.warning(f"Security alert triggered: {alert_id} - {alert.description}")
    
    def _has_active_alert(self, rule_id: str, user_id: str) -> bool:
        """Check if there's already an active alert for this rule and user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for unresolved alerts in last hour
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        
        cursor.execute('''
            SELECT COUNT(*) FROM security_alerts 
            WHERE alert_type = ? AND user_id = ? 
            AND resolved = FALSE AND timestamp > ?
        ''', (rule_id, user_id, one_hour_ago))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def _store_alert(self, alert: SecurityAlert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT INTO security_alerts 
                (alert_id, alert_type, severity, user_id, description, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (alert.alert_id, alert.alert_type, alert.severity, alert.user_id,
                  alert.description, alert.timestamp.isoformat(), json.dumps(alert.details)))
            conn.commit()
            self.active_alerts[alert.alert_id] = alert
        finally:
            conn.close()
    
    def _send_alert_notification(self, alert: SecurityAlert):
        """Send alert notification via email"""
        if not self.alert_recipients or not self.smtp_username:
            monitoring_logger.warning("Email notifications not configured")
            return
        
        try:
            subject = f"HIPAA Security Alert - {alert.severity} - {alert.alert_type}"
            
            body = f"""
HIPAA Security Alert

Alert ID: {alert.alert_id}
Severity: {alert.severity}
Type: {alert.alert_type}
User: {alert.user_id}
Time: {alert.timestamp.isoformat()}

Description: {alert.description}

Details:
{json.dumps(alert.details, indent=2)}

This is an automated alert from the HIPAA compliance monitoring system.
Please investigate immediately.
            """
            
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send to all recipients
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            
            for recipient in self.alert_recipients:
                if recipient.strip():
                    msg['To'] = recipient.strip()
                    server.send_message(msg)
                    del msg['To']
            
            server.quit()
            monitoring_logger.info(f"Alert notification sent: {alert.alert_id}")
            
        except Exception as e:
            monitoring_logger.error(f"Failed to send alert notification: {e}")
    
    def _analyze_patterns(self):
        """Analyze patterns for advanced threat detection"""
        try:
            # Analyze user behavior patterns
            self._analyze_user_patterns()
            
            # Check for coordinated attacks
            self._check_coordinated_attacks()
            
            # Analyze time-based patterns
            self._analyze_temporal_patterns()
            
        except Exception as e:
            monitoring_logger.error(f"Pattern analysis error: {e}")
    
    def _analyze_user_patterns(self):
        """Analyze individual user behavior patterns"""
        conn = sqlite3.connect("data/hipaa_audit.db")
        cursor = conn.cursor()
        
        # Get user activity in last 24 hours
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        
        cursor.execute('''
            SELECT user_id, COUNT(*) as activity_count
            FROM audit_log 
            WHERE timestamp > ? AND phi_accessed = TRUE
            GROUP BY user_id
            HAVING activity_count > 100
        ''', (yesterday,))
        
        high_activity_users = cursor.fetchall()
        conn.close()
        
        # Alert on unusually high activity
        for user_id, count in high_activity_users:
            if not self._has_active_alert("high_activity_pattern", user_id):
                alert = SecurityAlert(
                    alert_id=f"high_activity_{user_id}_{int(time.time())}",
                    alert_type="high_activity_pattern",
                    severity="MEDIUM",
                    user_id=user_id,
                    description=f"Unusually high PHI access activity: {count} accesses in 24 hours",
                    timestamp=datetime.utcnow(),
                    details={'activity_count': count, 'threshold': 100}
                )
                self._store_alert(alert)
                self._send_alert_notification(alert)
    
    def _check_coordinated_attacks(self):
        """Check for coordinated attack patterns"""
        conn = sqlite3.connect("data/hipaa_audit.db")
        cursor = conn.cursor()
        
        # Check for multiple users with failed access in short time
        five_minutes_ago = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
        
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) as unique_users
            FROM audit_log 
            WHERE timestamp > ? AND action LIKE '%failed%'
        ''', (five_minutes_ago,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] >= 5:  # 5 or more users with failures
            alert = SecurityAlert(
                alert_id=f"coordinated_attack_{int(time.time())}",
                alert_type="coordinated_attack",
                severity="CRITICAL",
                user_id="multiple_users",
                description=f"Potential coordinated attack: {result[0]} users with failures in 5 minutes",
                timestamp=datetime.utcnow(),
                details={'affected_users': result[0], 'time_window': 5}
            )
            self._store_alert(alert)
            self._send_alert_notification(alert)
    
    def _analyze_temporal_patterns(self):
        """Analyze time-based access patterns"""
        # This could include detecting unusual access times,
        # access frequency anomalies, etc.
        pass
    
    def _cleanup_old_events(self):
        """Clean up old events from memory"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for key in list(self.event_counters.keys()):
            events = self.event_counters[key]
            # Remove events older than 24 hours
            while events and events[0] < cutoff_time:
                events.popleft()
            
            # Remove empty counters
            if not events:
                del self.event_counters[key]
    
    def get_active_alerts(self, severity_filter: str = None) -> List[SecurityAlert]:
        """Get active security alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT alert_id, alert_type, severity, user_id, description, timestamp, details
            FROM security_alerts 
            WHERE resolved = FALSE
        '''
        params = []
        
        if severity_filter:
            query += ' AND severity = ?'
            params.append(severity_filter)
        
        query += ' ORDER BY timestamp DESC'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in results:
            alert = SecurityAlert(
                alert_id=row[0],
                alert_type=row[1],
                severity=row[2],
                user_id=row[3],
                description=row[4],
                timestamp=datetime.fromisoformat(row[5]),
                details=json.loads(row[6])
            )
            alerts.append(alert)
        
        return alerts
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve a security alert"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                UPDATE security_alerts 
                SET resolved = TRUE, resolved_at = ?, resolved_by = ?
                WHERE alert_id = ?
            ''', (datetime.utcnow().isoformat(), resolved_by, alert_id))
            
            conn.commit()
            resolved = conn.total_changes > 0
            
            if resolved and alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                
                # Log resolution
                self.audit_logger.log_phi_access(
                    resolved_by, 'security_alert_resolved', 'monitoring_system',
                    details={'alert_id': alert_id}
                )
                
            return resolved
            
        finally:
            conn.close()
    
    def get_security_dashboard(self) -> Dict:
        """Get security dashboard data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get alert statistics
        cursor.execute('''
            SELECT severity, COUNT(*) 
            FROM security_alerts 
            WHERE resolved = FALSE
            GROUP BY severity
        ''')
        
        alert_counts = dict(cursor.fetchall())
        
        # Get recent alert trends
        last_24h = (datetime.utcnow() - timedelta(days=1)).isoformat()
        cursor.execute('''
            SELECT alert_type, COUNT(*) 
            FROM security_alerts 
            WHERE timestamp > ?
            GROUP BY alert_type
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''', (last_24h,))
        
        alert_trends = cursor.fetchall()
        
        conn.close()
        
        return {
            'active_alerts': alert_counts,
            'total_active': sum(alert_counts.values()),
            'alert_trends_24h': alert_trends,
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global monitoring instance
security_monitor = HIPAASecurityMonitor()

def start_hipaa_monitoring():
    """Start HIPAA security monitoring"""
    security_monitor.start_monitoring()

def stop_hipaa_monitoring():
    """Stop HIPAA security monitoring"""
    security_monitor.stop_monitoring()

if __name__ == "__main__":
    # Test monitoring system
    print("Testing HIPAA Security Monitoring System...")
    
    # Start monitoring
    security_monitor.start_monitoring()
    
    print("Monitoring started. Checking dashboard...")
    time.sleep(2)
    
    # Get dashboard
    dashboard = security_monitor.get_security_dashboard()
    print(f"Dashboard: {json.dumps(dashboard, indent=2)}")
    
    # Simulate some events for testing
    print("Simulating security events...")
    
    # Stop monitoring
    security_monitor.stop_monitoring()
    print("Monitoring stopped.")