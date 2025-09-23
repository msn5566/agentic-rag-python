"""
Secure document store for managing uploaded files with security controls.
"""
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import shutil
import logging
from src.config import settings
from src.core.monitoring import security_logger, ThreatLevel

logger = logging.getLogger(__name__)


class SecureDocumentStore:
    """Secure document storage with access controls and audit trails."""
    
    def __init__(self):
        self.storage_dir = Path("uploads")
        self.metadata_dir = Path("storage/document_metadata")
        self.quarantine_dir = Path("storage/quarantine")
        
        # Create directories
        for directory in [self.storage_dir, self.metadata_dir, self.quarantine_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.authorized_users = set()
        self.access_log = []
    
    def authorize_user(self, user_id: str) -> None:
        """Authorize a user for document store access."""
        self.authorized_users.add(user_id)
        security_logger.log_security_event({
            "timestamp": datetime.utcnow(),
            "event_type": "user_authorized",
            "threat_level": ThreatLevel.LOW,
            "user_id": user_id,
            "ip_address": "internal",
            "query": None,
            "response": None,
            "metadata": {"action": "document_store_authorization"},
            "risk_score": 0.0
        })
    
    def _check_access(self, user_id: str, operation: str) -> bool:
        """Check if user is authorized for the operation."""
        if user_id not in self.authorized_users:
            security_logger.log_threat(
                threat_type="unauthorized_document_access",
                threat_level=ThreatLevel.HIGH,
                user_id=user_id,
                ip_address="internal",
                details={
                    "operation": operation,
                    "authorized_users": len(self.authorized_users)
                },
                risk_score=0.8
            )
            return False
        
        # Log authorized access
        self.access_log.append({
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "operation": operation
        })
        
        return True
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Save file metadata."""
        metadata_file = self.metadata_dir / f"{file_path.stem}.json"
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _load_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load file metadata."""
        metadata_file = self.metadata_dir / f"{file_path.stem}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {file_path}: {e}")
            return None
    
    def store_file(self, file_content: bytes, filename: str, user_id: str, 
                   security_validated: bool = False, risk_score: float = 0.0) -> Dict[str, Any]:
        """Store file securely with metadata."""
        if not self._check_access(user_id, "store_file"):
            raise PermissionError(f"User {user_id} not authorized for document store access")
        
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        file_path = self.storage_dir / safe_filename
        
        # Handle duplicate filenames
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            suffix = original_path.suffix
            file_path = self.storage_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        try:
            # Write file
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Create metadata
            metadata = {
                "original_filename": filename,
                "stored_filename": file_path.name,
                "file_size": len(file_content),
                "file_hash": file_hash,
                "uploaded_by": user_id,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "security_validated": security_validated,
                "risk_score": risk_score,
                "mime_type": None,  # To be filled by validation
                "indexed": False,
                "quarantined": False
            }
            
            # Save metadata
            self._save_metadata(file_path, metadata)
            
            # Log successful storage
            security_logger.app_logger.info(
                f"File stored successfully: {safe_filename} by user {user_id}"
            )
            
            security_logger.log_security_event({
                "timestamp": datetime.utcnow(),
                "event_type": "file_stored",
                "threat_level": ThreatLevel.LOW,
                "user_id": user_id,
                "ip_address": "internal",
                "query": None,
                "response": None,
                "metadata": {
                    "filename": safe_filename,
                    "file_size": len(file_content),
                    "risk_score": risk_score
                },
                "risk_score": risk_score
            })
            
            return {
                "success": True,
                "file_path": str(file_path),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to store file {filename}: {e}")
            security_logger.log_threat(
                threat_type="file_storage_error",
                threat_level=ThreatLevel.MEDIUM,
                user_id=user_id,
                ip_address="internal",
                details={
                    "filename": filename,
                    "error": str(e)
                },
                risk_score=0.4
            )
            raise
    
    def quarantine_file(self, file_path: Path, reason: str, user_id: str) -> None:
        """Move file to quarantine."""
        if not self._check_access(user_id, "quarantine_file"):
            raise PermissionError(f"User {user_id} not authorized for document store access")
        
        try:
            quarantine_path = self.quarantine_dir / file_path.name
            shutil.move(str(file_path), str(quarantine_path))
            
            # Update metadata
            metadata = self._load_metadata(file_path) or {}
            metadata.update({
                "quarantined": True,
                "quarantine_reason": reason,
                "quarantine_timestamp": datetime.utcnow().isoformat(),
                "quarantined_by": user_id
            })
            
            # Save updated metadata
            self._save_metadata(quarantine_path, metadata)
            
            security_logger.log_threat(
                threat_type="file_quarantined",
                threat_level=ThreatLevel.HIGH,
                user_id=user_id,
                ip_address="internal",
                details={
                    "filename": file_path.name,
                    "reason": reason,
                    "quarantine_path": str(quarantine_path)
                },
                risk_score=0.7
            )
            
        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path}: {e}")
            raise
    
    def get_file_info(self, filename: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get file information."""
        if not self._check_access(user_id, "get_file_info"):
            raise PermissionError(f"User {user_id} not authorized for document store access")
        
        file_path = self.storage_dir / filename
        
        if not file_path.exists():
            # Check quarantine
            quarantine_path = self.quarantine_dir / filename
            if quarantine_path.exists():
                file_path = quarantine_path
            else:
                return None
        
        metadata = self._load_metadata(file_path)
        if metadata:
            metadata["current_path"] = str(file_path)
            metadata["exists"] = file_path.exists()
        
        return metadata
    
    def list_files(self, user_id: str, include_quarantined: bool = False) -> List[Dict[str, Any]]:
        """List all files with metadata."""
        if not self._check_access(user_id, "list_files"):
            raise PermissionError(f"User {user_id} not authorized for document store access")
        
        files = []
        
        # List regular files
        for file_path in self.storage_dir.iterdir():
            if file_path.is_file():
                metadata = self._load_metadata(file_path)
                if metadata:
                    metadata["current_path"] = str(file_path)
                    metadata["status"] = "active"
                    files.append(metadata)
        
        # List quarantined files if requested
        if include_quarantined:
            for file_path in self.quarantine_dir.iterdir():
                if file_path.is_file():
                    metadata = self._load_metadata(file_path)
                    if metadata:
                        metadata["current_path"] = str(file_path)
                        metadata["status"] = "quarantined"
                        files.append(metadata)
        
        return files
    
    def delete_file(self, filename: str, user_id: str, permanent: bool = False) -> bool:
        """Delete file and its metadata."""
        if not self._check_access(user_id, "delete_file"):
            raise PermissionError(f"User {user_id} not authorized for document store access")
        
        file_path = self.storage_dir / filename
        quarantine_path = self.quarantine_dir / filename
        
        # Find the file
        target_path = None
        if file_path.exists():
            target_path = file_path
        elif quarantine_path.exists():
            target_path = quarantine_path
        
        if not target_path:
            return False
        
        try:
            # Load metadata before deletion
            metadata = self._load_metadata(target_path)
            
            # Delete file
            target_path.unlink()
            
            # Delete metadata
            metadata_file = self.metadata_dir / f"{target_path.stem}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            security_logger.log_security_event({
                "timestamp": datetime.utcnow(),
                "event_type": "file_deleted",
                "threat_level": ThreatLevel.LOW,
                "user_id": user_id,
                "ip_address": "internal",
                "query": None,
                "response": None,
                "metadata": {
                    "filename": filename,
                    "permanent": permanent,
                    "original_metadata": metadata
                },
                "risk_score": 0.1
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {e}")
            security_logger.log_threat(
                threat_type="file_deletion_error",
                threat_level=ThreatLevel.MEDIUM,
                user_id=user_id,
                ip_address="internal",
                details={
                    "filename": filename,
                    "error": str(e)
                },
                risk_score=0.3
            )
            return False
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove directory traversal attempts
        filename = os.path.basename(filename)
        
        # Replace dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        # Ensure filename is not empty
        if not filename or filename.startswith('.'):
            filename = f"file_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return filename
    
    def get_storage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._check_access(user_id, "get_storage_stats"):
            raise PermissionError(f"User {user_id} not authorized for document store access")
        
        try:
            active_files = list(self.storage_dir.iterdir())
            quarantined_files = list(self.quarantine_dir.iterdir())
            
            total_size = sum(f.stat().st_size for f in active_files if f.is_file())
            quarantine_size = sum(f.stat().st_size for f in quarantined_files if f.is_file())
            
            return {
                "active_files": len([f for f in active_files if f.is_file()]),
                "quarantined_files": len([f for f in quarantined_files if f.is_file()]),
                "total_size_bytes": total_size,
                "quarantine_size_bytes": quarantine_size,
                "authorized_users": len(self.authorized_users),
                "access_log_entries": len(self.access_log)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
