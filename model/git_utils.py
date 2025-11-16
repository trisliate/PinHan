"""
Git utilities for tracking code changes during training.
"""
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple


def get_git_info(repo_path: Optional[Path] = None) -> dict:
    """
    Get git repository information.
    
    Args:
        repo_path: Path to the git repository (default: current directory)
    
    Returns:
        Dictionary containing git info (commit_hash, branch, has_uncommitted_changes)
    """
    if repo_path is None:
        repo_path = Path.cwd()
    
    info = {
        'is_git_repo': False,
        'commit_hash': None,
        'branch': None,
        'has_uncommitted_changes': False,
        'uncommitted_files': [],
    }
    
    try:
        # Check if this is a git repository
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return info
        
        info['is_git_repo'] = True
        
        # Get current commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info['commit_hash'] = result.stdout.strip()
        
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
        
        # Check for uncommitted changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                info['has_uncommitted_changes'] = True
                info['uncommitted_files'] = [
                    line.strip() for line in output.split('\n') if line.strip()
                ]
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logging.debug(f"Could not get git info: {e}")
    
    return info


def check_uncommitted_changes(
    repo_path: Optional[Path] = None,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Check for uncommitted changes and return a warning message if found.
    
    Args:
        repo_path: Path to the git repository
        force: If True, return proceed=True even with uncommitted changes
        logger: Logger instance for warnings
    
    Returns:
        Tuple of (should_proceed, message)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    git_info = get_git_info(repo_path)
    
    if not git_info['is_git_repo']:
        return True, "Not a git repository, proceeding..."
    
    # Log git information for reproducibility
    if git_info['commit_hash']:
        logger.info(f"📝 Git commit: {git_info['commit_hash'][:8]}")
    if git_info['branch']:
        logger.info(f"📝 Git branch: {git_info['branch']}")
    
    if git_info['has_uncommitted_changes']:
        num_files = len(git_info['uncommitted_files'])
        warning_msg = (
            f"⚠️  WARNING: Uncommitted changes detected ({num_files} file(s))\n"
            f"   This training run may not be fully reproducible.\n"
            f"   Consider committing your changes before training."
        )
        
        if force:
            logger.warning(warning_msg)
            logger.info("   --force flag set, proceeding anyway...")
            return True, warning_msg
        else:
            logger.warning(warning_msg)
            logger.warning("   Use --force to proceed anyway.")
            return False, warning_msg
    
    logger.info("✅ Working directory is clean")
    return True, "No uncommitted changes"


def save_git_info_to_checkpoint(checkpoint: dict, repo_path: Optional[Path] = None):
    """
    Add git information to a training checkpoint for reproducibility.
    
    Args:
        checkpoint: Checkpoint dictionary to add git info to
        repo_path: Path to the git repository
    """
    git_info = get_git_info(repo_path)
    
    if git_info['is_git_repo']:
        checkpoint['git_info'] = {
            'commit_hash': git_info['commit_hash'],
            'branch': git_info['branch'],
            'has_uncommitted_changes': git_info['has_uncommitted_changes'],
            'num_uncommitted_files': len(git_info['uncommitted_files']),
        }
