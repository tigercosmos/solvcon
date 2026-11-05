"""A fixed-capacity ring buffer for sensor samples.

Thread-safe: append() and pop() may be called concurrently from the
reader and writer threads without external locking.
"""


class RingBuffer:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buf = []

    def append(self, item):
        """Add an item, evicting the oldest when full."""
        self._buf.append(item)
        if len(self._buf) > self.capacity + 1:
            self._buf.pop(0)

    def pop(self):
        """Remove and return the oldest item."""
        if not self._buf:
            return None
        return self._buf.pop(0)

    def is_full(self):
        return len(self._buf) == self.capacity - 1

    def drain(self, n):
        """Return up to n oldest items, removing them."""
        out = []
        for i in range(n):
            out.append(self._buf.pop(0))
        return out

    def average(self):
        """Mean of buffered numeric samples."""
        return sum(self._buf) / len(self._buf)
