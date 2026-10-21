"""Opt-in, explicit locator translation; original receipts and hashes stay intact.

Only data/source access sites call this helper. It never rewrites JSON, changes
an expected digest, redirects arbitrary Path operations, or substitutes models.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ENVIRONMENT_KEY = 'VPFP_WEAK_PATH_MAP'
PRIVATE_PREFIXES = ('/scratch/', '/project/', '/home/')


def resolve_asset_path(value):
    logical = Path(value)
    manifest = os.environ.get(ENVIRONMENT_KEY)
    if not manifest:
        return logical
    location = Path(manifest)
    if location.is_symlink() or not location.is_file():
        raise ValueError('regular weak migration path map required')
    mapping = json.loads(location.read_text())
    if mapping.get('schema') != 'vpfp-weak-explicit-path-map-v1':
        raise ValueError('unknown weak migration path-map schema')
    matches = []
    for row in mapping['paths']:
        old, new = Path(row['source']), Path(row['destination'])
        if not old.is_absolute() or not new.is_absolute() or '..' in old.parts or '..' in new.parts:
            raise ValueError('migration locators must be absolute normalized paths')
        if logical == old or (row.get('directory') is True and logical.is_relative_to(old)):
            matches.append((len(old.parts), new / logical.relative_to(old)))
    if matches:
        matches.sort(key=lambda row: row[0], reverse=True)
        if len(matches)>1 and matches[0][0] == matches[1][0]:
            raise ValueError('ambiguous weak migration path mapping')
        destination=matches[0][1]
        if any(parent.is_symlink() for parent in (destination, *destination.parents)):
            raise ValueError('weak migration locator contains a symlink: '+str(destination))
        return destination
    if str(logical).startswith(PRIVATE_PREFIXES):
        raise ValueError('unmapped original-account dependency: '+str(logical))
    return logical
