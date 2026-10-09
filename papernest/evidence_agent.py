"""Compatibility import for the chapter Evidence Agent.

The implementation lives in :mod:`papernest.evidence`; this module keeps the
role-oriented name available to integrations that import ``evidence_agent``.
"""

from .evidence import EvidenceAgent, build_evidence_bundle

__all__ = ["EvidenceAgent", "build_evidence_bundle"]
