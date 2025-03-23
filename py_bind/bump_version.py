#!/usr/bin/env python3
"""Version bumping utility for optv package."""

import re
import sys
from pathlib import Path

def bump_version(version_type='patch'):
    """
    Bump the version number.
    version_type: 'major', 'minor', or 'patch'
    """
    version_file = Path('optv/version.py')
    content = version_file.read_text()
    
    # Extract current version
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Version pattern not found in version.py")
    
    current_version = match.group(1)
    major, minor, patch = map(int, current_version.split('.'))
    
    # Update version numbers
    if version_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif version_type == 'minor':
        minor += 1
        patch = 0
    elif version_type == 'patch':
        patch += 1
    else:
        raise ValueError("Invalid version type. Use 'major', 'minor', or 'patch'")
    
    new_version = f"{major}.{minor}.{patch}"
    
    # Update version.py
    new_content = re.sub(
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        f'__version__ = "{new_version}"',
        content
    )
    version_file.write_text(new_content)
    
    print(f"Version bumped from {current_version} to {new_version}")

if __name__ == '__main__':
    version_type = sys.argv[1] if len(sys.argv) > 1 else 'patch'
    bump_version(version_type)