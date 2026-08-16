import os
import shutil
import hashlib
from typing import Protocol, List, Optional
from pathlib import Path


class StorageBackend(Protocol):
    """Protocol defining the interface for pluggable persistent storage."""
    
    def write(self, path: str, content: bytes) -> str:
        """Writes content to the storage backend and returns a URI/path."""
        ...
        
    def read(self, path: str) -> bytes:
        """Reads content from the storage backend."""
        ...
        
    def exists(self, path: str) -> bool:
        """Checks if a file exists in the storage backend."""
        ...
        
    def delete(self, path: str) -> bool:
        """Deletes a file from the storage backend."""
        ...

    def list_files(self, prefix: str) -> List[str]:
        """Lists files matching a prefix."""
        ...


class LocalStorage:
    """Local file system implementation of StorageBackend."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _resolve(self, path: str) -> Path:
        # Ensure path is relative and within base_dir (basic security check)
        clean_path = path.lstrip('/')
        resolved = (self.base_dir / clean_path).resolve()
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    def write(self, path: str, content: bytes, overwrite: bool = False) -> str:
        target = self._resolve(path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Immutable storage: file already exists at {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'wb') as f:
            f.write(content)
        return str(target)
        
    def read(self, path: str) -> bytes:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        with open(target, 'rb') as f:
            return f.read()
            
    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()
        
    def delete(self, path: str) -> bool:
        target = self._resolve(path)
        if target.exists():
            target.unlink()
            return True
        return False

    def list_files(self, prefix: str) -> List[str]:
        prefix_path = self.base_dir / prefix.lstrip('/')
        if not prefix_path.parent.exists():
            return []
            
        results = []
        for p in self.base_dir.rglob('*'):
            if p.is_file():
                rel_path = str(p.relative_to(self.base_dir)).replace('\\', '/')
                if rel_path.startswith(prefix.lstrip('/')):
                    results.append(rel_path)
        return results


def calculate_hash(content: bytes) -> str:
    """Utility to calculate SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()
