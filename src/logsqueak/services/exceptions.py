"""Custom exceptions for Logsqueak services."""


class FileModifiedError(Exception):
    """Raised when a file is modified during an atomic write operation.

    This exception indicates that the file changed between the initial
    read and the final write, which could lead to data loss if the write
    were to proceed.

    Attributes:
        path: Path to the file that was modified
        message: Human-readable error message
    """

    def __init__(self, path: str, message: str = "File was modified during write operation"):
        """Initialize FileModifiedError.

        Args:
            path: Path to the file that was modified
            message: Human-readable error message
        """
        self.path = path
        self.message = message
        super().__init__(f"{message}: {path}")
