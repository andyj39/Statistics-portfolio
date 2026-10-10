import numpy as np
import scipy.stats as st
from collections import deque
from traffic_wait_replications import Simulation, SCENARIOS, ci

class EmergencySim(Simulation):
    """Adds emergency vehicles (0.5% of arrivals). With override=True, the light
    multiplier is set to its maximum (4) while an emergency vehicle is queued:
    red shortens to 1/4 of base, green stretches to 4x base."""
    def __init__(self, override=False, p_emerg=0.005, clamp=True, seed=0, **kw):
        super().__init__(seed=seed, **kw)
        self.override, self.p, self.clamp = override, p_emerg, clamp
        self.rng_e = np.random.default_rng(seed + 10_000)   # separate stream for emergency draws
        self.flags, self.n_emerg = deque(), 0
        self.e_waits, self.n_waits = [], []

    def light_rate(self):
        if self.override and self.n_emerg > 0:
            return 4
        return super().light_rate()

    def arrival(self):
        super().arrival()
        e = bool(self.rng_e.random() < self.p)
        self.flags.append(e)
        self.n_emerg += e

    def depart(self):
        e = self.flags.popleft()
        self.n_emerg -= e
        super().depart()
        (self.e_waits if e else self.n_waits).append(self.waits[-1])

    def red_step(self):
        if self.t_arrival <= self.t_green:
            self.arrival()
        else:
            self.light_state = 'green'
        t_g = self.t_red + self.mu / self.light_rate()
        self.t_green = max(self.clock, t_g) if self.clamp else t_g   # never schedule green in the past

MODES = {"fixed light": dict(dynamic=False, override=False),
         "dynamic":     dict(dynamic=True,  override=False),
         "dynamic + emergency override": dict(dynamic=True, override=True)}

def run(R=30, scenario=None, clamp=True):
    out = {}
    for name, m in MODES.items():
        ew, nw = [], []
        for seed in range(R):
            s = EmergencySim(seed=seed, clamp=clamp, **m, **scenario).run()
            assert min(s.waits) > -1e-9, "negative wait"
            ew.append(np.mean(s.e_waits) * 60); nw.append(np.mean(s.n_waits) * 60)
        out[name] = (np.array(ew), np.array(nw))
    return out

if __name__ == "__main__":
    sc = SCENARIOS["calibrated 2015 (6/12 per min, steps of 3 cars)"]
    for clamp in (False, True):
        print(f"\n== calibrated scenario, 30 runs x 5 days, clamp={clamp}")
        res = run(30, sc, clamp)
        for name, (e, n) in res.items():
            print(f"{name:30s} emergency wait {ci(e):>12s} s | ordinary cars {ci(n):>12s} s")
        d = res["dynamic"][0] - res["dynamic + emergency override"][0]
        print(f"override saves emergency vehicles {ci(d)} s vs dynamic ({100*d.mean()/res['dynamic'][0].mean():.0f}%)")
