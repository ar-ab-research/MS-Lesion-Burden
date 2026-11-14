"""One base class, so the command line can report any expected failure the same way."""


class PipelineError(ValueError):
    """Something the user can fix: a bad config, a bad table, a missing or unusable file."""
