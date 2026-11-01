class ToolError(Exception):
    """A tool returned a declared failure code (slot_unavailable, slot_taken, ...)."""
    def __init__(self, code, msg=""):
        super().__init__(code)
        self.code = code
        self.msg = msg


class Transient503(ToolError):
    def __init__(self):
        super().__init__("transient_503")


class ToolTimeout(Exception):
    """The call timed out. The outcome is UNKNOWN (it may or may not have happened)."""


class ApprovalRequired(Exception):
    """A mutating call was attempted without a valid approval covering it."""


class BudgetExceeded(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class SimulatedCrash(BaseException):
    """Test-only: simulates the process dying. BaseException so no `except Exception` swallows it."""
