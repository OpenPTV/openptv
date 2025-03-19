#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

def read_version():
    with open("version.txt") as f:
        return f.read().strip()

def write_version(version):
    with open("version.txt", "w") as f:
        f.write(f"{version}\n")

def bump_version(version_str, bump_type):
    major, minor, patch = map(int, version_str.split('.'))
    
    if bump_type == 'major':
        return f"{major + 1}.0.0"
    elif bump_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif bump_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"
    else:
        return version_str

def main():
    parser = argparse.ArgumentParser(description='Bump project version')
    parser.add_argument('bump_type', choices=['major', 'minor', 'patch'],
                      help='Which version component to bump')
    args = parser.parse_args()

    current_version = read_version()
    new_version = bump_version(current_version, args.bump_type)
    write_version(new_version)
    print(f"Version bumped from {current_version} to {new_version}")

if __name__ == '__main__':
    main()