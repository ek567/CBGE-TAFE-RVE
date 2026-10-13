from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
ADAPTATION_REVISION_ID = 'CBGE_TAFE_RVE_GEOMETRY_V2_2'
DEFAULT_PROFILE_ID = 'THREE_CONSTRAINTS'
PARENT_BASELINE_ID = 'CBGE_TAFE_RVE_GEOMETRY_ENGINE'

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _safe_member(root: Path, member: str) -> Path:
    if not isinstance(member, str):
        raise RuntimeError('Identity manifest member must be text')
    parsed = PurePosixPath(member)
    if parsed.is_absolute() or '..' in parsed.parts or '\\' in member or (':' in member):
        raise RuntimeError(f'Unsafe identity manifest member: {member!r}')
    path = (root / member).resolve()
    if root not in path.parents:
        raise RuntimeError(f'Identity member escapes its bound root: {member!r}')
    return path

def read_current_adaptation_identity() -> dict:
    from geometry_acceptance_retry import verify_source_identity
    return verify_source_identity()
