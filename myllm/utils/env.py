import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Returns the portable project root path using the following precedence:
    1. PROJECT_ROOT environment variable
    2. Git repository root (.git directory)
    3. Current working directory fallback
    """
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"]).resolve()
        
    cwd = Path.cwd().resolve()
    current = cwd
    
    # Search upwards for .git
    for _ in range(5):
        if (current / ".git").is_dir():
            return current
        current = current.parent
        
    return cwd
