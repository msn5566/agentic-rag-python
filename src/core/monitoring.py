"""
Comprehensive monitoring and logging system for RAG application security.
"""
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from src.config import settings


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    user_id: Optional[str]
    ip_address: str
    query: Optional[str]
    response: Optional[str]
    metadata: Dict[str, Any]
    risk_score: float


class SecurityLogger:
    """Enhanced security logging system."""
    
    def __init__(self):
        self.setup_logging()
        self.events = deque(maxlen=10000)  # Keep last 10k events in memory
        
    def setup_logging(self):
        """Configure security logging."""
        os.makedirs("logs", exist_ok=True)
        
        # Security logger
        self.security_logger = logging.getLogger("security")
        self.security_logger.setLevel(logging.INFO)
        
        # Security file handler
        security_handler = logging.FileHandler("logs/security.log")
        security_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        security_handler.setFormatter(security_formatter)
        self.security_logger.addHandler(security_handler)
        
        # Application logger
        self.app_logger = logging.getLogger("app")
        self.app_logger.setLevel(getattr(logging, settings.log_level))
        
        app_handler = logging.FileHandler(settings.log_file)
        app_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        app_handler.setFormatter(app_formatter)
        self.app_logger.addHandler(app_handler)
    
    def log_security_event(self, event):
        """Log a security event."""
        # Handle both SecurityEvent objects and dictionaries
        if isinstance(event, dict):
            # Convert dict to SecurityEvent-like structure
            from datetime import datetime
            event_data = {
                "timestamp": event.get("timestamp", datetime.utcnow()).isoformat() if hasattr(event.get("timestamp", datetime.utcnow()), 'isoformat') else str(event.get("timestamp", datetime.utcnow())),
                "event_type": event.get("event_type", "unknown"),
                "threat_level": event.get("threat_level", ThreatLevel.LOW).value if hasattr(event.get("threat_level", ThreatLevel.LOW), 'value') else str(event.get("threat_level", "low")),
                "user_id": event.get("user_id"),
                "ip_address": event.get("ip_address"),
                "risk_score": event.get("risk_score", 0.0),
                "metadata": event.get("metadata", {})
            }
            self.events.append(event)
        else:
            # Handle SecurityEvent objects
            self.events.append(event)
            event_data = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "threat_level": event.threat_level.value,
                "user_id": event.user_id,
                "ip_address": event.ip_address,
                "risk_score": event.risk_score,
                "metadata": event.metadata
        }
        
        # Don't log sensitive query/response data in security logs
        query = event.get("query") if isinstance(event, dict) else getattr(event, "query", None)
        response = event.get("response") if isinstance(event, dict) else getattr(event, "response", None)
        
        if query:
            event_data["query_hash"] = hashlib.sha256(str(query).encode()).hexdigest()[:16]
        if response:
            event_data["response_hash"] = hashlib.sha256(str(response).encode()).hexdigest()[:16]
        
        self.security_logger.info(json.dumps(event_data))
        
        # Alert on high/critical threats
        threat_level = event.get("threat_level") if isinstance(event, dict) else getattr(event, "threat_level", ThreatLevel.LOW)
        event_type = event.get("event_type") if isinstance(event, dict) else getattr(event, "event_type", "unknown")
        risk_score = event.get("risk_score") if isinstance(event, dict) else getattr(event, "risk_score", 0.0)
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.security_logger.error(f"HIGH THREAT DETECTED: {event_type} - Risk Score: {risk_score}")
    
    def log_query(self, user_id: str, ip_address: str, query: str, response: str, metadata: Dict[str, Any]):
        """Log a query interaction."""
        self.app_logger.info(f"Query from user {user_id} ({ip_address}): {query[:100]}...")
        
        # Create security event for query
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="query_processed",
            threat_level=ThreatLevel.LOW,
            user_id=user_id,
            ip_address=ip_address,
            query=query,
            response=response,
            metadata=metadata,
            risk_score=0.1
        )
        self.log_security_event(event)
    
    def log_threat(self, threat_type: str, threat_level: ThreatLevel, user_id: str, 
                   ip_address: str, details: Dict[str, Any], risk_score: float):
        """Log a security threat."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type=threat_type,
            threat_level=threat_level,
            user_id=user_id,
            ip_address=ip_address,
            query=details.get("query"),
            response=details.get("response"),
            metadata=details,
            risk_score=risk_score
        )
        self.log_security_event(event)


class AnomalyDetector:
    """Detects anomalous behavior patterns."""
    
    def __init__(self, logger: SecurityLogger):
        self.logger = logger
        self.user_patterns = defaultdict(lambda: {
            "query_count": 0,
            "last_queries": deque(maxlen=50),
            "query_times": deque(maxlen=100),
            "suspicious_patterns": 0
        })
        
    def analyze_query(self, user_id: str, ip_address: str, query: str) -> float:
        """Analyze query for anomalies and return risk score."""
        risk_score = 0.0
        threats_detected = []
        
        # Update user patterns
        user_data = self.user_patterns[user_id]
        user_data["query_count"] += 1
        user_data["last_queries"].append(query)
        user_data["query_times"].append(datetime.utcnow())
        
        # Check for prompt injection patterns
        injection_score = self._detect_prompt_injection(query)
        risk_score += injection_score
        if injection_score > 0.5:
            threats_detected.append("prompt_injection")
        
        # Check for data extraction attempts
        extraction_score = self._detect_data_extraction(query)
        risk_score += extraction_score
        if extraction_score > 0.3:
            threats_detected.append("data_extraction")
        
        # Check for rate limiting violations
        rate_score = self._check_rate_patterns(user_data)
        risk_score += rate_score
        if rate_score > 0.4:
            threats_detected.append("rate_abuse")
        
        # Check for query similarity (potential automated attacks)
        similarity_score = self._check_query_similarity(user_data["last_queries"])
        risk_score += similarity_score
        if similarity_score > 0.3:
            threats_detected.append("automated_queries")
        
        # Log threats if detected
        if threats_detected:
            threat_level = ThreatLevel.HIGH if risk_score > 0.7 else ThreatLevel.MEDIUM
            self.logger.log_threat(
                threat_type="anomaly_detected",
                threat_level=threat_level,
                user_id=user_id,
                ip_address=ip_address,
                details={
                    "query": query,
                    "threats": threats_detected,
                    "injection_score": injection_score,
                    "extraction_score": extraction_score,
                    "rate_score": rate_score,
                    "similarity_score": similarity_score
                },
                risk_score=risk_score
            )
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _detect_prompt_injection(self, query: str) -> float:
        """Detect potential prompt injection attempts."""
        injection_patterns = [
            r"ignore\s+(previous|above|all)\s+(instructions?|prompts?|rules?)",
            r"forget\s+(everything|all|previous)",
            r"you\s+are\s+now\s+a?",
            r"system\s*:\s*",
            r"assistant\s*:\s*",
            r"human\s*:\s*",
            r"\[INST\]|\[/INST\]",
            r"<\|system\|>|<\|user\|>|<\|assistant\|>",
            r"roleplay\s+as",
            r"pretend\s+(you\s+are|to\s+be)",
            r"act\s+as\s+(if\s+)?you\s+(are|were)",
            r"bypass\s+(safety|security|filter)",
            r"jailbreak",
            r"developer\s+mode",
            r"admin\s+mode"
        ]
        
        score = 0.0
        query_lower = query.lower()
        
        for pattern in injection_patterns:
            if re.search(pattern, query_lower):
                score += 0.3
        
        # Check for excessive special characters
        special_chars = len(re.findall(r'[<>{}[\]|\\]', query))
        if special_chars > 5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_data_extraction(self, query: str) -> float:
        """Detect attempts to extract sensitive data."""
        extraction_patterns = [
            r"show\s+me\s+(all|every)",
            r"list\s+(all|every)",
            r"what\s+(data|information)\s+do\s+you\s+have",
            r"dump\s+(database|data|everything)",
            r"export\s+(all|data)",
            r"give\s+me\s+(access|admin|root)",
            r"password|secret|key|token|credential",
            r"confidential|classified|internal",
            r"personal\s+information|pii|ssn|social\s+security",
            r"credit\s+card|bank\s+account|financial",
            r"api\s+key|access\s+token"
        ]
        
        score = 0.0
        query_lower = query.lower()
        
        for pattern in extraction_patterns:
            if re.search(pattern, query_lower):
                score += 0.2
        
        return min(score, 1.0)
    
    def _check_rate_patterns(self, user_data: Dict) -> float:
        """Check for suspicious rate patterns."""
        now = datetime.utcnow()
        recent_queries = [
            t for t in user_data["query_times"] 
            if (now - t).total_seconds() < 60
        ]
        
        # More than 20 queries per minute is suspicious
        if len(recent_queries) > 20:
            return 0.5
        
        # More than 10 queries per minute is concerning
        if len(recent_queries) > 10:
            return 0.3
        
        return 0.0
    
    def _check_query_similarity(self, queries: deque) -> float:
        """Check for suspiciously similar queries."""
        if len(queries) < 5:
            return 0.0
        
        recent_queries = list(queries)[-10:]  # Check last 10 queries
        
        # Simple similarity check based on common words
        similarity_count = 0
        for i in range(len(recent_queries) - 1):
            for j in range(i + 1, len(recent_queries)):
                words1 = set(recent_queries[i].lower().split())
                words2 = set(recent_queries[j].lower().split())
                
                if len(words1) > 0 and len(words2) > 0:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    if similarity > 0.8:  # 80% similarity
                        similarity_count += 1
        
        # If more than 30% of query pairs are highly similar
        total_pairs = len(recent_queries) * (len(recent_queries) - 1) // 2
        if total_pairs > 0 and similarity_count / total_pairs > 0.3:
            return 0.4
        
        return 0.0


class OutputFilter:
    """Filters and sanitizes LLM outputs."""
    
    def __init__(self, logger: SecurityLogger):
        self.logger = logger
        
    def filter_response(self, response: str, user_id: str, ip_address: str) -> str:
        """Filter response for sensitive information."""
        original_response = response
        
        # Enhanced PII detection and masking
        response = self._mask_enhanced_pii(response)
        
        # Remove potential system information
        response = self._remove_system_info(response)
        
        # Check for data leakage
        leakage_score = self._detect_data_leakage(response)
        
        if leakage_score > 0.5:
            self.logger.log_threat(
                threat_type="data_leakage_detected",
                threat_level=ThreatLevel.HIGH,
                user_id=user_id,
                ip_address=ip_address,
                details={
                    "original_response": original_response,
                    "filtered_response": response,
                    "leakage_score": leakage_score
                },
                risk_score=leakage_score
            )
        
        return response
    
    def _mask_enhanced_pii(self, text: str) -> str:
        """Enhanced PII masking."""
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     lambda m: f"{m.group()[0]}***@{m.group().split('@')[1]}", text)
        
        # Phone numbers (various formats)
        text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                     r'***-***-**\3', text)
        
        # Social Security Numbers
        text = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '***-**-****', text)
        
        # Credit card numbers
        text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '**** **** **** ****', text)
        
        # IP addresses
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '***.***.***.**', text)
        
        # API keys and tokens (long alphanumeric strings)
        text = re.sub(r'\b[A-Za-z0-9]{20,}\b', '[TOKEN_MASKED]', text)
        
        return text
    
    def _remove_system_info(self, text: str) -> str:
        """Remove system information from responses."""
        # Remove file paths
        text = re.sub(r'[A-Za-z]:\\[^\\]+(?:\\[^\\]+)*', '[PATH_REMOVED]', text)
        text = re.sub(r'/[^/\s]+(?:/[^/\s]+)*', '[PATH_REMOVED]', text)
        
        # Remove internal URLs
        text = re.sub(r'http://(?:localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):\d+[^\s]*', 
                     '[INTERNAL_URL_REMOVED]', text)
        
        return text
    
    def _detect_data_leakage(self, response: str) -> float:
        """Detect potential data leakage in response."""
        score = 0.0
        
        # Check for common sensitive patterns
        sensitive_patterns = [
            r'password\s*[:=]\s*\w+',
            r'secret\s*[:=]\s*\w+',
            r'key\s*[:=]\s*\w+',
            r'token\s*[:=]\s*\w+',
            r'database\s+connection',
            r'internal\s+server',
            r'confidential',
            r'classified'
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.2
        
        return min(score, 1.0)


# Global instances
security_logger = SecurityLogger()
anomaly_detector = AnomalyDetector(security_logger)
output_filter = OutputFilter(security_logger)
