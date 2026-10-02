"""Cubic and quintic interpolation of s(t) along a straight line, through via points.

s is the distance travelled along the line (0 -> L). The line itself is
p(t) = p_start + s(t) * u,  u = unit vector p_start -> p_end.

Method (Craig, "Introduction to Robotics", ch. 7 -- polynomials with via points):
  * Knots: s_0=0, s_1..s_n (the 3-4 intermediate points), s_last=L.
  * Nominal knot times come from a normalised rest-to-rest quintic, so knots
    are spread in time like a smooth motion (dense where the robot is slow).
  * CUBIC: clamped cubic spline -- one cubic per segment, v = 0 at both ends,
    via velocities chosen so position, velocity AND acceleration are
    continuous at the via points. Price: with only 4 coefficients per segment
    the acceleration at t=0 and t=T cannot also be 0 -> it jumps from 0 to a
    finite value when the motion starts/stops (infinite jerk there).
  * QUINTIC: one quintic per segment matching the same via positions,
    velocities and accelerations (taken from the spline), plus a = 0 at both
    ends -> acceleration continuous everywhere and zero at start/stop.
  * Time scaling: all knot times are multiplied by k. Velocity scales with 1/k
    and acceleration with 1/k^2, so
        k = max(v_peak / v_max, sqrt(a_peak / a_max))
    gives the fastest profile of that shape that still respects BOTH limits
    (the binding limit is reached exactly, the other is below its limit).
"""
import numpy as np


def _quintic_blend(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def _invert_blend(frac):
    """tau in [0,1] such that the normalised quintic reaches `frac`."""
    grid = np.linspace(0.0, 1.0, 20001)
    return float(np.interp(frac, _quintic_blend(grid), grid))


class ViaPointProfile:
    def __init__(self, s_knots, t_knots, kind):
        if kind not in ('cubic', 'quintic'):
            raise ValueError(kind)
        self.kind = kind
        self.s = np.asarray(s_knots, float)
        self.t = np.asarray(t_knots, float)
        n = len(self.s)
        h = np.diff(self.t)

        # Via velocities: clamped cubic spline (v=0 at both ends, acceleration
        # continuous at every via point) -> tridiagonal system in v_1..v_{n-2}
        v = np.zeros(n)
        if n > 2:
            A = np.zeros((n - 2, n - 2))
            b = np.zeros(n - 2)
            for r, i in enumerate(range(1, n - 1)):
                A[r, r] = 2.0 * (1.0 / h[i - 1] + 1.0 / h[i])
                if r > 0:
                    A[r, r - 1] = 1.0 / h[i - 1]
                if r < n - 3:
                    A[r, r + 1] = 1.0 / h[i]
                b[r] = 3.0 * ((self.s[i] - self.s[i - 1]) / h[i - 1] ** 2
                              + (self.s[i + 1] - self.s[i]) / h[i] ** 2)
            v[1:-1] = np.linalg.solve(A, b)

        # Via accelerations = those of the cubic spline; 0 at both ends (quintic only)
        a = np.zeros(n)
        if kind == 'quintic':
            for i in range(1, n - 1):
                c = self._cubic(self.s[i], self.s[i + 1], v[i], v[i + 1], h[i])
                a[i] = 2 * c[2]
        self.v_knots, self.a_knots = v, a

        self.coeffs = []
        for i in range(n - 1):
            if kind == 'cubic':
                self.coeffs.append(self._cubic(self.s[i], self.s[i + 1], v[i], v[i + 1], h[i]))
            else:
                self.coeffs.append(self._quintic(self.s[i], self.s[i + 1], v[i], v[i + 1],
                                                 a[i], a[i + 1], h[i]))

    @staticmethod
    def _cubic(s0, s1, v0, v1, T):
        c0, c1 = s0, v0
        c2 = 3 * (s1 - s0) / T ** 2 - (2 * v0 + v1) / T
        c3 = -2 * (s1 - s0) / T ** 3 + (v0 + v1) / T ** 2
        return np.array([c0, c1, c2, c3, 0.0, 0.0])

    @staticmethod
    def _quintic(s0, s1, v0, v1, a0, a1, T):
        h = s1 - s0
        c3 = (20 * h - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
        c4 = (-30 * h + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
        c5 = (12 * h - 6 * (v1 + v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)
        return np.array([s0, v0, a0 / 2, c3, c4, c5])

    @property
    def duration(self):
        return float(self.t[-1])

    def evaluate(self, tq):
        """Returns s, v, a, jerk at times tq."""
        tq = np.atleast_1d(np.asarray(tq, float))
        idx = np.clip(np.searchsorted(self.t, tq, side='right') - 1, 0, len(self.coeffs) - 1)
        out = np.zeros((4, len(tq)))
        for k, (ti, seg) in enumerate(zip(tq, idx)):
            c = self.coeffs[seg]
            x = min(max(ti - self.t[seg], 0.0), self.t[seg + 1] - self.t[seg])
            out[0, k] = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4 + c[5] * x ** 5
            out[1, k] = c[1] + 2 * c[2] * x + 3 * c[3] * x ** 2 + 4 * c[4] * x ** 3 + 5 * c[5] * x ** 4
            out[2, k] = 2 * c[2] + 6 * c[3] * x + 12 * c[4] * x ** 2 + 20 * c[5] * x ** 3
            out[3, k] = 6 * c[3] + 24 * c[4] * x + 60 * c[5] * x ** 2
        return out

    def time_of_s(self, s_query, dt=1e-3):
        """Inverse s -> t (s(t) is monotonic for these profiles)."""
        tg = np.arange(0.0, self.duration + dt, dt)
        tg[-1] = self.duration
        sg = np.maximum.accumulate(self.evaluate(tg)[0])
        sg_u, idx = np.unique(sg, return_index=True)
        return np.interp(s_query, sg_u, tg[idx])

    def peaks(self, n_per_segment=400):
        """Peak |v|, |a|, |jerk|, evaluating every segment on its own closed
        interval so the one-sided values at the knots (where the cubic's
        acceleration jumps) are included."""
        pk = np.zeros(3)
        for i, c in enumerate(self.coeffs):
            x = np.linspace(0.0, self.t[i + 1] - self.t[i], n_per_segment)
            v = c[1] + 2 * c[2] * x + 3 * c[3] * x ** 2 + 4 * c[4] * x ** 3 + 5 * c[5] * x ** 4
            a = 2 * c[2] + 6 * c[3] * x + 12 * c[4] * x ** 2 + 20 * c[5] * x ** 3
            j = 6 * c[3] + 24 * c[4] * x + 60 * c[5] * x ** 2
            pk = np.maximum(pk, [np.max(np.abs(v)), np.max(np.abs(a)), np.max(np.abs(j))])
        return float(pk[0]), float(pk[1]), float(pk[2])


def design_profile(length, v_max, a_max, kind, via_fractions=(0.20, 0.50, 0.80)):
    """Fastest via-point profile of type `kind` over `length` metres within limits."""
    fr = [0.0] + list(via_fractions) + [1.0]
    s_knots = [length * f for f in fr]
    t_knots = [_invert_blend(f) for f in fr]        # nominal duration = 1 s
    base = ViaPointProfile(s_knots, t_knots, kind)
    v_pk, a_pk, _ = base.peaks()
    k = max(v_pk / v_max, np.sqrt(a_pk / a_max))
    return ViaPointProfile(s_knots, [t * k for t in t_knots], kind)
