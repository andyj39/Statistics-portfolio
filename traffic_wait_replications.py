import numpy as np
import scipy.stats as st
from collections import deque

SCENARIOS = {
    "original (5/10/15 per min, steps of 5 cars)": dict(rates=[5,5,5,10,15,15,5,5,5,5], step=5),
    "calibrated 2015 (6/12 per min, steps of 3 cars)": dict(rates=[6,6,6,12,12,6,6,6,6,6], step=3),
}

class Simulation:
    """Your model, with: no printing, a fixed/dynamic switch, per-car waits,
    and (fix=True) two corrections described in the notes."""
    def __init__(self, mu=1, t_car=2/60, maxtime=3000, dynamic=True,
                 rates=None, step=5, seed=None, fix=True):
        self.rng = np.random.default_rng(seed)
        self.mu, self.t_car, self.maxtime = mu, t_car, maxtime
        self.dynamic, self.rates, self.step, self.fix = dynamic, rates, step, fix
        self.clock, self.light_state = 0.0, 'red'
        self.num_in_system, self.t_red = 0, 0.0
        self.t_arrival = self.interarrival()
        self.t_green = mu
        self.t_depart = mu + t_car
        self.arrival_times, self.waits, self.queue_at_green = deque(), [], []

    def rate(self, t):
        return self.rates[int((t % 600) // 60)]

    def interarrival(self):
        return self.rng.exponential(1 / self.rate(self.clock))

    def light_rate(self):
        if not self.dynamic:
            return 1
        return min(4, 1 + 0.5 * (self.num_in_system // self.step))

    def arrival(self):
        self.num_in_system += 1
        self.arrival_times.append(self.clock)
        self.t_arrival = self.clock + self.interarrival()

    def depart(self):
        self.num_in_system -= 1
        self.waits.append(self.clock - self.arrival_times.popleft())
        self.t_depart = self.clock + self.t_car

    def red_step(self):
        if self.t_arrival <= self.t_green:
            self.arrival()
        else:
            self.light_state = 'green'
        # FIX 3: never schedule the next green in the past (stops the clock rewinding)
        self.t_green = max(self.clock, self.t_red + self.mu / self.light_rate())

    def green_phase(self):
        self.t_red = self.clock + self.mu * self.light_rate()
        if self.fix:                                   # FIX 2
            self.t_depart = self.clock + self.t_car
        self.queue_at_green.append(self.num_in_system)
        while self.clock < self.t_red:
            if self.fix and self.num_in_system == 0:   # FIX 1
                t_event = self.t_arrival
            else:
                t_event = min(self.t_arrival, self.t_depart)
            self.clock = t_event
            if self.clock >= self.t_red:
                break
            if t_event == self.t_arrival or (not self.fix and self.num_in_system == 0):
                was_empty = self.num_in_system == 0
                self.arrival()
                if self.fix and was_empty:
                    self.t_depart = self.clock + self.t_car
            else:
                self.depart()
            self.t_red = self.t_green + self.mu * self.light_rate()
        self.clock = self.t_red
        self.t_green = self.clock + self.mu / self.light_rate()
        self.t_depart = self.t_green + self.t_car
        self.light_state = 'red'

    def run(self):
        while self.clock <= self.maxtime:
            self.clock = min(self.t_arrival, self.t_green)
            if self.light_state == 'red':
                self.red_step()
            if self.light_state == 'green':
                self.green_phase()
        return self

def replicate(dynamic, R, scenario, fix=True):
    mw, mq, zero, left = [], [], [], []
    for seed in range(R):
        s = Simulation(dynamic=dynamic, seed=seed, fix=fix, **scenario).run()
        w = np.array(s.waits)
        mw.append(w.mean() * 60)                 # seconds
        mq.append(np.mean(s.queue_at_green))
        zero.append(np.mean(w < 1e-9))
        left.append(s.num_in_system)
    return np.array(mw), np.array(mq), np.array(zero), np.array(left)

def ci(x):
    h = st.t.ppf(0.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.1f} ± {h:.1f}"

if __name__ == "__main__":
    R = 30
    for name, sc in SCENARIOS.items():
        print("\n==", name)
        fw, fq, _, fl = replicate(False, R, sc)
        dw, dq, _, dl = replicate(True,  R, sc)
        d = fw - dw
        print(f"mean wait/car (s)  fixed {ci(fw)} | dynamic {ci(dw)}")
        print(f"reduction (s)      {ci(d)}  = {100*d.mean()/fw.mean():.0f}%")
        print(f"mean queue at green fixed {ci(fq)} | dynamic {ci(dq)}")
        print(f"cars still queued at end (mean) fixed {fl.mean():.1f} | dynamic {dl.mean():.1f}")
    print("\n== Effect of the two corrections (calibrated scenario, mean wait/car in s)")
    sc = SCENARIOS["calibrated 2015 (6/12 per min, steps of 3 cars)"]
    for fixflag in (False, True):
        fw, *_ = replicate(False, R, sc, fix=fixflag)
        dw, _, dz, _ = replicate(True, R, sc, fix=fixflag)
        print(f"corrections={fixflag}: fixed {ci(fw)} | dynamic {ci(dw)} | share of zero-wait cars (dynamic) {dz.mean():.0%}")
