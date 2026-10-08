"""Exception hierarchy.

Everything the harness raises on purpose derives from ``HarnessError`` so callers
can tell a policy decision apart from a bug.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all intentional harness errors."""


class Denied(HarnessError):
    """The sandbox refused an action (path escape, disallowed command, ...).

    A denial is *expected* behaviour: it is returned to the model as an error
    tool result and recorded in the audit log, never swallowed.
    """


class ToolInputError(HarnessError):
    """The model called a tool with arguments that don't match its schema."""


class CheckpointError(HarnessError):
    """A checkpoint is missing, unreadable, or from an incompatible version."""


class AuditIntegrityError(HarnessError):
    """The audit log's hash chain does not verify."""
