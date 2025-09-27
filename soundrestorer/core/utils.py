class RobustAverager:
    def __init__(self, trim_frac: float = 0.05):
        self.trim = float(trim_frac)
        self.vals = []
    def add(self, v: float): self.vals.append(float(v))
    def mean(self) -> float:
        if not self.vals: return 0.0
        vs = sorted(self.vals)
        k = int(len(vs) * self.trim)
        core = vs[k: len(vs)-k] if len(vs) - 2*k > 0 else vs
        return sum(core) / max(1, len(core))