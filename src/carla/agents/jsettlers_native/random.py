"""java.util.Random's 48-bit generator for reproducible reference choices."""


class JavaRandom:
    def __init__(self, seed):
        self.state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def bits(self, count):
        self.state = (self.state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.state >> (48 - count)

    def next_int(self):
        value = self.bits(32)
        return value if value < 1 << 31 else value - (1 << 32)

    def index(self, count):
        return abs(self.next_int()) % count
