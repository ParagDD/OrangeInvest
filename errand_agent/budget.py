"""Hard caps on model calls, tool calls and wall-clock time for a run."""
from .errors import BudgetExceeded


class Budget:
    def __init__(self, clock, max_model_calls=30, max_tool_calls=45, max_wall_s=300.0):
        self.clock, self.max_model_calls, self.max_tool_calls, self.max_wall_s = clock, max_model_calls, max_tool_calls, max_wall_s
        self.t0 = clock.now()
        self.model_calls = 0
        self.tool_calls = 0

    def begin_turn(self):
        """Wall-clock bounds *active work in a turn*; time the user spends thinking between turns is free."""
        self.t0 = self.clock.now()

    def _time(self):
        if self.clock.now() - self.t0 > self.max_wall_s:
            raise BudgetExceeded("wall_clock")

    def charge_model(self):
        self._time()
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("model_calls")
        self.model_calls += 1

    def charge_tool(self):
        self._time()
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("tool_calls")
        self.tool_calls += 1
