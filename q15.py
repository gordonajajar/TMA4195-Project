
import os
import numpy as np
from scipy.signal import place_poles
import matplotlib.pyplot as plt
 
from q9 import M, T, T_END, WINDOW, d
from q12 import A, B, B_d, POLES, gain


OBS_POLES = [-2.0, -2.5, -3.0, -3.5]
DT = 1e-3
X0 = np.array([0.5, 0.0, 0.0, 0.0])
XHAT0 = np.zeros(4)
NOISE_REL = 0.05    # Relative noise set at 5% of RMS

SENSORS = {
    "y = x":   (np.array([1.0, 0.0, 0.0, 0.0]), 0.0, 0.0),
    "y = xdd": (A[1], B[1], B_d[1]),
}

# Similar to gain from q12, 
def observer_gain(C, poles):
    L = place_poles(A.T, C.reshape(-1, 1), poles).gain_matrix.T[:, 0]
    achieved = np.sort_complex(np.linalg.eigvals(A - np.outer(L, C)))
    assert np.allclose(achieved, np.sort_complex(poles)), achieved
    return L


def no_disturbance(t):
    return 0.0


def simulate(K, sensor, L, dist, t_end, noise_std=0.0, seed=42):
    """Fixed-step RK4 on the linear plant + observer.
 
    sensor = None gives the full-state reference u = -K X.
    The noise sample is held constant over each step (sampled sensor).
    Returns t, X (4 x N), X_hat (4 x N), u (N).
    """
    rng = np.random.default_rng(seed)
    n = int(round(t_end / DT))
    t = np.linspace(0.0, n * DT, n + 1)
    Z = np.empty((8, n + 1))
    Z[:, 0] = np.concatenate([X0, XHAT0])
    u_out = np.empty(n + 1)

    # If we have no sensor, 
    if sensor is None:
        C, D, Dd = np.zeros(4), 0.0, 0.0
        L = np.zeros(4)
    else:
        C, D, Dd = sensor


    def rhs(tt, z, noise):
        X, Xh = z[:4], z[4:]
        dd = dist(tt)
        u = -K @ (X if sensor is None else Xh)
        y = C @ X + D * u + Dd * dd + noise
        dX = A @ X + B * u + B_d * dd
        dXh = A @ Xh + B * u + L * (y - C @ Xh - D * u)
        return np.concatenate([dX, dXh]), u

    # RK4
    for i in range(n):
        noise = noise_std * rng.standard_normal() if noise_std > 0 else 0.0
        z, ti = Z[:, i], t[i]
        k1, u_out[i] = rhs(ti, z, noise)
        k2, _ = rhs(ti + DT / 2, z + DT / 2 * k1, noise)
        k3, _ = rhs(ti + DT / 2, z + DT / 2 * k2, noise)
        k4, _ = rhs(ti + DT, z + DT * k3, noise)
        Z[:, i + 1] = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    
    u_out[-1] = -K @ (Z[:4, -1] if sensor is None else Z[4:, -1])
    return t, Z[:4], Z[4:], u_out

def measured_signal(sensor, X, u, t, dist):
    C, D, Dd = sensor
    return C @ X + D * u + Dd * np.array([dist(tt) for tt in t])


def noise_levels(K):
    t, X, _, u = simulate(K, None, None, d, T_END)
    window = t >= T_END - WINDOW
    levels = {}
    for name, sensor in SENSORS.items():
        y = measured_signal(sensor, X, u, t, d)[window]
        levels[name] = NOISE_REL * np.sqrt(np.mean(y**2))
    return levels

# FIGURES

def scenario_figure(K, gains, dist, noisy, levels, title, fname, t_end):
    t, X_ref, _, _ = simulate(K, None, None, dist, t_end)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(t, X_ref[0], "k--", lw=1.0, label="full-state $u = -KX$")
    window = t >= t_end - min(WINDOW, t_end / 2)   # 'late' = steady part

    print(f"\n{title}")
    print(f"  {'controller':<34s} {'max|x| (late)':>13s} {'rms x (late)':>12s}"
          f" {'mean|e| (late)':>14s} {'max|u| (late)':>13s}")
    print(f"  {'full-state reference':<34s} {np.max(np.abs(X_ref[0][window])):13.3e}"
          f" {np.sqrt(np.mean(X_ref[0][window]**2)):12.3e} {'-':>14s}"
          f" {np.max(np.abs((-K @ X_ref)[window])):13.3e}")
    
    for name, sensor in SENSORS.items():
        std = levels[name] if noisy else 0.0
        t, X, Xh, u = simulate(K, sensor, gains[name], dist, t_end, std)
        err = np.linalg.norm(X - Xh, axis=0)
        axes[0].plot(t, X[0], lw=1.0, label=name)
        axes[1].semilogy(t, err, lw=1.0, label=name)
        print(f"  {name:<34s} {np.max(np.abs(X[0][window])):13.3e}"
              f" {np.sqrt(np.mean(X[0][window]**2)):12.3e}"
              f" {np.mean(err[window]):14.3e} {np.max(np.abs(u[window])):13.3e}")
        
    axes[0].set_ylabel("$x(t)$")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title, fontsize=10)
    axes[1].set_ylabel(r"$\|X(t) - \hat X(t)\|$")
    axes[1].set_xlabel("rescaled time $t$")
    axes[1].legend(fontsize=8)
    axes[1].grid(which="both", color="0.9")
    fig.tight_layout()
    fig.savefig(f"figures/{fname}")
    plt.close(fig)
 
 
# MAIN
 
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
 
    K = gain(POLES)
    print(f"controller poles {POLES}")
    print(f"K = {np.array2string(K, precision=4)}")
 
    gains = {name: observer_gain(sensor[0], OBS_POLES)
             for name, sensor in SENSORS.items()}
    print(f"observer poles {OBS_POLES}")
    for name, L in gains.items():
        print(f"  L ({name}) = {np.array2string(L, precision=4)}")
 
    # Disturbance forcing of the error, e' = (A - LC)e + (B_d - L Dd) d
    for name, (C, D, Dd) in SENSORS.items():
        print(f"  B_d - L*Dd ({name}) = "
              f"{np.array2string(B_d - gains[name] * Dd, precision=4)}")
 
    levels = noise_levels(K)
    for name, s in levels.items():
        print(f"  noise std ({name}) = {s:.3e}  ({NOISE_REL:.0%} of signal rms)")
 
    T_SHORT = 20.0
    scenario_figure(K, gains, no_disturbance, False, levels,
                    "1. $d = 0$, no noise", "q15_1_ideal.pdf", T_SHORT)
    scenario_figure(K, gains, no_disturbance, True, levels,
                    "2. $d = 0$, measurement noise", "q15_2_noise.pdf", T_SHORT)
    scenario_figure(K, gains, d, False, levels,
                    "3. Q9 disturbance, no noise", "q15_3_disturbance.pdf", T_END)
    scenario_figure(K, gains, d, True, levels,
                    "4. Q9 disturbance and noise", "q15_4_both.pdf", T_END)
    print("\nFigures written to ./figures/")