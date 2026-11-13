"""
Question 9: open-loop disturbance cancellation

Three parts:
    1. reduced model (f = u): apply u_d, check that x stays at 0
    2. full actuator model (ef' + f = u): apply the same u_d for several eps
       and see how actuator lag affects the cancellation
    3. initial layer (for fun): actuator starting at f(0) = 0 instead of
       f(0) = u_d(0)

Everything is in rescaled time t (refer to Section 3 of report).
Output into figures/
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# plot settings, same as q7
plt.rc("font", family="serif", size=10)
plt.rc("mathtext", fontset="cm")
plt.rc("axes", titlesize=10, grid=True,
       prop_cycle=plt.cycler(color=["teal", "firebrick", "darkgoldenrod", "indigo"]))
plt.rc("grid", linestyle=":", color="0.7")
plt.rc("legend", fontsize=9)

# PARAMETERS
# Same values as the later control sections, so z_d and u_d can be reused
M = m = k = c = k_a = c_a = 1.0
T = np.sqrt(M / k)       # time scale, T = 1
beta = 0.2               # same as q7

# Disturbance d(t) = P [cos(OM1 t) + 1/2 cos(OM2 t)]
P = 0.1
OM1, OM2 = 1.0, 2.3

# Actuator time constants to test (eps = tau_a / T)
EPSILONS = [0.001, 0.01, 0.05, 0.1, 0.2]

# Simulation settings
# Both tones repeat together every 20 pi (OM2/OM1 = 23 in 10), so the steady-state
# measure uses the last 20 pi of the simulation, one full period of d.
T_END = 150.0
WINDOW = 20 * np.pi
RTOL, ATOL = 1e-9, 1e-12  # same as q7, strict tolerances


# DESIRED TRAJECTORY AND CONTROL (see formulas from the report, Q9 setup)
# I write everything in terms of the variables so the code can be easily reused or changed for different parameters.


# Disturbance d
def d(t):
    return P * (np.cos(OM1 * t) + 0.5 * np.cos(OM2 * t))


# Absorber motion z_d that keeps x = 0
def z_d(t):
    return -(T**2 / m) * P * (np.cos(OM1 * t) / OM1**2
                              + 0.5 * np.cos(OM2 * t) / OM2**2)


def z_d_dot(t):
    return (T**2 / m) * P * (np.sin(OM1 * t) / OM1
                             + 0.5 * np.sin(OM2 * t) / OM2)


# Open-loop control from Q8
def u_d(t):
    return d(t) + (c_a / T) * z_d_dot(t) + k_a * z_d(t)


# Initial state on the desired trajectory: x = x' = 0, z = z_d(0), z' = z_d'(0)
# z_d(0) is computed from the formula, so it is exact up to rounding
Y0 = [0.0, 0.0, z_d(0.0), z_d_dot(0.0)]


# MODELS


# Forces in the rescaled equations (eq:resc-x, eq:resc-z), shared by both models
# Returns the accelerations x'' and z'' for a given actuator force f
def accelerations(t, x, v, z, w, f):
    a_x = (d(t) - f - (c / T) * v - (c_a / T) * (v - w)
           - k * x - beta * x**3 - k_a * (x - z)) / (M / T**2)
    a_z = (f - (c_a / T) * (w - v) - k_a * (z - x)) / (m / T**2)
    return a_x, a_z


# Reduced model
# State y = (x, v, z, w), v = x', w = z'
def rhs_reduced(t, y):
    x, v, z, w = y
    a_x, a_z = accelerations(t, x, v, z, w, u_d(t)) # f = u_d, instant
    return [v, a_x, w, a_z]


# Full model: the actuator force f lags, ef' + f = u_d.
# State y = (x, v, z, w, f)
def rhs_full(t, y, eps):
    x, v, z, w, f = y
    a_x, a_z = accelerations(t, x, v, z, w, f)
    f_dot = (u_d(t) - f) / eps
    return [v, a_x, w, a_z, f_dot]


# SIMULATION FUNCTIONS


# Reduced model. Not stiff, so the explicit method DOP853 is used (same as Q7)
def simulate_reduced(rtol=RTOL, atol=ATOL, n_out=15001):
    t_eval = np.linspace(0.0, T_END, n_out)
    sol = solve_ivp(rhs_reduced, (0.0, T_END), Y0, method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    assert sol.success, sol.message
    return sol.t, sol.y[0], sol.y[2]


# Full model. Stiff when eps is small (f changes on a time scale eps, the rest
# on a time scale 1), so the implicit method Radau is used
# f0 is the actuator force at t = 0, by default use u_d(0) (actuator already pushing)
# Returns t, x, z, f and the number of right-hand-side evaluations (solver effort)
def simulate_full(eps, f0=None, method="Radau", rtol=RTOL, atol=ATOL,
                  t_end=T_END, n_out=15001):
    if f0 is None:
        f0 = u_d(0.0)
    t_eval = np.linspace(0.0, t_end, n_out)
    sol = solve_ivp(rhs_full, (0.0, t_end), Y0 + [f0], method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol, args=(eps,))
    assert sol.success, sol.message
    return sol.t, sol.y[0], sol.y[2], sol.y[4], sol.nfev


# Vibration measure: max |x| over the last WINDOW (one full period of d),
# after the transient has died out
def steady_amplitude(t, x):
    window = t >= T_END - WINDOW
    return np.max(np.abs(x[window]))


# FIGURES


# 1. Reduced model: the absorber follows z_d and the primary mass stays at 0
def reduced_figure():
    t, x, z = simulate_reduced()
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t, d(t), label="disturbance $d$")
    axes[0].plot(t, u_d(t), label="control $u_d$")
    axes[0].set_ylabel("force")
    axes[0].set_ylim(-0.2, 0.27)
    axes[0].legend(loc="upper center", ncol=2)
    axes[1].plot(t, z, label="simulated $z$")
    axes[1].plot(t, z_d(t), "--", label="desired $z_d$")
    axes[1].set_ylabel("$z(t)$")
    axes[1].set_ylim(-0.15, 0.2)
    axes[1].legend(loc="upper center", ncol=2)
    axes[2].plot(t, x)
    axes[2].set_ylabel("$x(t)$")
    axes[2].set_xlabel("rescaled time $t$")
    axes[0].set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig("figures/q9_reduced.pdf")
    plt.close(fig)
    # printed numbers for report
    print("\nReduced model (f = u_d):")
    print(f"  max |x|       over 0 <= t <= {T_END:.0f}: {np.max(np.abs(x)):.1e}")
    print(f"  max |z - z_d| over 0 <= t <= {T_END:.0f}: {np.max(np.abs(z - z_d(t))):.1e}")


# 2. Full model: x(t) for several eps, and the steady-state amplitude versus eps
def full_figure():
    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    print("\nFull actuator model, f(0) = u_d(0):")
    print(f"  {'eps':>6s}  {'max|x| (steady)':>16s}  {'max|x| / eps':>12s}")
    amps = []
    for eps in EPSILONS:
        t, x, _, _, _ = simulate_full(eps)
        amp = steady_amplitude(t, x)
        amps.append(amp)
        print(f"  {eps:6.3f}  {amp:16.3e}  {amp / eps:12.4f}")
        # eps = 0.001 is too small to see on a linear scale, so leave it out here
        if eps >= 0.01:
            axes[0].plot(t, x, lw=0.9, label=fr"$\varepsilon = {eps}$")
    axes[0].set_xlim(0, 60)
    axes[0].set_xlabel("rescaled time $t$")
    axes[0].set_ylabel("$x(t)$")
    axes[0].legend(loc="upper right")
    # log-log plot: a straight line of slope 1 means amplitude proportional to eps, expected for small eps (see section 3)
    eps_arr = np.array(EPSILONS)
    axes[1].loglog(eps_arr, amps, "o-", label="simulation")
    axes[1].loglog(eps_arr, amps[2] / eps_arr[2] * eps_arr, "--", color="grey", lw=0.8,
                   label=r"slope 1 ($\propto \varepsilon$)")
    axes[1].set_xlabel(r"actuator time constant $\varepsilon$")
    axes[1].set_ylabel(r"steady-state max$|x|$")
    axes[1].legend()
    axes[1].grid(which="minor", color="0.9")
    fig.tight_layout()
    fig.savefig("figures/q9_full.pdf")
    plt.close(fig)


# Extra
# 3. Initial layer: actuator switched on at t = 0 (f(0) = 0) versus already
#    pushing (f(0) = u_d(0)), for eps = 0.1. See how fast initial layer dies
def initial_layer_figure():
    eps = 0.1
    t1, x1, _, f1, _ = simulate_full(eps, t_end=20.0, n_out=4001)
    t0, x0, _, f0, _ = simulate_full(eps, f0=0.0, t_end=20.0, n_out=4001) # actuator switched on at t = 0
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5))
    # force only near t = 0, where the layer is
    early = t1 <= 1.0
    axes[0].plot(t1[early], u_d(t1[early]), "k:", label="command $u_d$")
    axes[0].plot(t1[early], f1[early], label="$f$, $f(0) = u_d(0)$")
    axes[0].plot(t0[early], f0[early], label="$f$, $f(0) = 0$")
    axes[0].set_ylabel("actuator force")
    axes[0].legend()
    axes[1].plot(t1, x1, label="$f(0) = u_d(0)$")
    axes[1].plot(t0, x0, label="$f(0) = 0$")
    axes[1].set_ylabel("$x(t)$")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("rescaled time $t$")
    fig.tight_layout()
    fig.savefig("figures/q9_initial_layer.pdf")
    plt.close(fig)
    # how far apart the two runs are, compared with the size of x itself
    print(f"\nInitial layer (eps = {eps}): gap between the two runs")
    print(f"  {'window':>8s}  {'max|gap|':>9s}  {'max|x|':>9s}")
    for a, b in [(0, 3), (3, 6), (6, 10), (10, 15), (15, 20)]:
        w = (t1 >= a) & (t1 < b)
        gap = np.max(np.abs(x0[w] - x1[w]))
        print(f"  {a:3d}-{b:<4d}  {gap:9.2e}  {np.max(np.abs(x1[w])):9.2e}")


# 4. Accuracy checks. See report
def accuracy_checks():
    # Check 1: the reduced model should keep x at 0 exactly, so max |x| (printed
    # by reduced_figure) shows the size of the numerical error directly. Expect noise in order of tolerances

    # Check 2: rerun the full model with 100x stricter tolerances.
    # If the amplitude doesn't change, the solver error is negligible.
    print("\nAccuracy check: tolerance tightened 100x (full model, Radau)")
    for eps in (0.001, 0.1):
        t, x, _, _, _ = simulate_full(eps)
        t2, x2, _, _, _ = simulate_full(eps, rtol=1e-11, atol=1e-14)
        a1, a2 = steady_amplitude(t, x), steady_amplitude(t2, x2)
        print(f"  eps={eps:5.3f}: {a1:.8e}  vs  {a2:.8e}  "
              f"rel.diff={abs(a1 - a2) / a2:.1e}")

    # Check 3: stiffness. Compare the solver effort (right-hand-side evaluations)
    # of the explicit DOP853 and the implicit Radau on a short run.
    # An explicit method needs far more evaluations when eps is small.
    print("\nStiffness: right-hand-side evaluations on 0 <= t <= 20")
    for eps in (0.1, 0.001):
        *_, n_exp = simulate_full(eps, method="DOP853", t_end=20.0, n_out=201)
        *_, n_imp = simulate_full(eps, method="Radau", t_end=20.0, n_out=201)
        print(f"  eps={eps:5.3f}: DOP853 {n_exp:7d}   Radau {n_imp:7d}")


# MAIN
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print(f"z_d(0) = {z_d(0.0):.6f},  z_d'(0) = {z_d_dot(0.0):.6f},  "
          f"u_d(0) = {u_d(0.0):.6f}")
    reduced_figure()
    full_figure()
    initial_layer_figure()
    accuracy_checks()
    print("\nFigures written to ./figures/")
