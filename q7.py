"""
Question 7: passive vibration absorber

Three cases:
    1. no absorber, decoupled z
    2. tuned absorber
    3. mistuned absorber

Output into figures/
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# plot settings: font to match the report
plt.rc("font", family="serif", size=10)
plt.rc("mathtext", fontset="cm")
plt.rc("axes", titlesize=10, grid=True,
       prop_cycle=plt.cycler(color=["teal", "firebrick", "darkgoldenrod", "indigo"]))
plt.rc("grid", linestyle=":", color="0.7")
plt.rc("legend", fontsize=9)

# PARAMETERS
M, k = 1.0, 1.0
OMEGA0 = np.sqrt(k / M)  # natural frequency of primary mass
c = 0.04                 # damping constant of primary mass
beta = 0.2
P = 0.02
m = 0.1
ZETA_A = 0.05            # damping ratio of absorber


# SIMULATION FUNCTIONS


# For an absorber tuned to freq, return spring and damping constants k_a, c_a
def absorber(freq):
    k_a = m * freq**2
    c_a = 2 * ZETA_A * np.sqrt(k_a * m)
    return k_a, c_a


# Used by figure functions to obtain k_a, c_a for each case
CASES = {
    "(i) no absorber": (0.0, 0.0),
    "(ii) tuned absorber": absorber(OMEGA0),
    "(iii) mistuned absorber (+30 %)": absorber(1.3 * OMEGA0),
}


# Simulation settings
# https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.integrate.DOP853.html
T_END = 600.0             # long enough for steady state
N_MEASURE = 20            # number of forcing periods used for vibration measure
RTOL, ATOL = 1e-9, 1e-12  # tolerances. This is much stricter than default
METHOD = "DOP853"         # Explicit ODE solver. Precise, efficient for non-stiff


# This function takes in time t and state y, returns y'
def rhs(t, y, k_a, c_a, Omega, b):
    # state y = (x, v, z, w), v = x', w = z'
    x, v, z, w = y
    d = P * np.cos(Omega * t)
    a_x = (d - c * v - c_a * (v - w) - k * x - b * x**3 - k_a * (x - z)) / M
    a_z = (-c_a * (w - v) - k_a * (z - x)) / m
    return [v, a_x, w, a_z]


# This function returns time and state (x,z) as an array of n_out output points over time period 0 - T_END
# Uses beta=0.2 and starts from rest unless stated otherwise.
def simulate(k_a, c_a, Omega, b=beta, rtol=RTOL, atol=ATOL, n_out=20000,
             y0=(0.0, 0.0, 0.0, 0.0)):
    # split time evenly into n_out points, into array t_eval
    t_eval = np.linspace(0.0, T_END, n_out)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    sol = solve_ivp(rhs, (0.0, T_END), list(y0),
                    method=METHOD, t_eval=t_eval,
                    rtol=rtol, atol=atol,
                    args=(k_a, c_a, Omega, b))
    # alert if solver fails
    assert sol.success, sol.message
    return sol.t, sol.y[0], sol.y[2]


# Input: time array t, displacement array x, forcing frequency Omega
# Output: vibration measure as max |x| and RMS of x, of last N_MEASURE forcing periods
def steady_measures(t, x, Omega):
    # We want to measure after steady state, so we take the last N_MEASURE
    # forcing periods of the simulation. One period is 2pi/Omega, so 
    # we are looking at T_END - N_MEASURE * 2pi / Omega.
    window = t >= T_END - N_MEASURE * 2 * np.pi / Omega
    return np.max(np.abs(x[window])), np.sqrt(np.mean(x[window]**2))


# Exact steady state amplitudes |X|, |Z| for linear system (b=0). Used for comparing with simulation results for accuracy check.
def linear_amplitude(k_a, c_a, Omega):
    i = 1j
    K = np.array([
        [k + k_a - M * Omega**2 + i * Omega * (c + c_a), -k_a - i * Omega * c_a],
        [-k_a - i * Omega * c_a, k_a - m * Omega**2 + i * Omega * c_a],
    ])
    X, Z = np.linalg.solve(K, [P, 0.0])
    return abs(X), abs(Z)


# FIGURES


# 1. Time series with forcing at primary resonance OMEGA0, for all three cases
# Plots x(t) for each case in its own panel, and prints the steady-state
# measures (max |x|, RMS of x, absorber stroke, which is its max |z - x|)
def time_series_figure():
    Omega = OMEGA0
    # 3 panels stacked vertically, sharing the time axis
    fig, axes = plt.subplots(len(CASES), 1, figsize=(7, 6.5), sharex=True)
    print(f"\nSteady state at Omega = {Omega:.2f}  (beta = {beta})")
    print(f"{'case':34s} {'max|x|':>9s} {'RMS x':>9s} {'max|z-x|':>10s}")
    # loop over panels and cases together
    for ax, (name, (k_a, c_a)) in zip(axes, CASES.items()):
        t, x, z = simulate(k_a, c_a, Omega)
        amp, rms = steady_measures(t, x, Omega)
        # absorber stroke = spring compression z - x, over the same window.
        # nan (not a number) for no absorber, since there is no spring.
        win = t >= T_END - N_MEASURE * 2 * np.pi / Omega
        stroke = np.max(np.abs(z[win] - x[win])) if k_a > 0 else float("nan")
        print(f"{name:34s} {amp:9.4f} {rms:9.4f} {stroke:10.4f}")
        ax.plot(t, x, lw=0.8)
        ax.set_ylabel("$x(t)$")
        ax.set_title(f"{name}: steady-state max$|x|$ = {amp:.3f}")
        # same y-range in every panel
        ax.set_ylim(-0.7, 0.7)
    axes[-1].set_xlabel("time $t$")
    fig.tight_layout()
    fig.savefig("figures/q7_time_series.pdf")
    plt.close(fig)


# 2. Frequency response: steady-state amplitude versus different forcing Omega
# Solid lines: simulated max |x| (beta = 0.2), one full simulation per Omega
# Dashed lines: exact linear amplitude (beta = 0), as a reference for the effect of beta
def frequency_response_figure():
    # Test 81 frequencies, spacing 0.01, so peak locations are accurate to ~0.01
    Omegas = np.linspace(0.6, 1.4, 81)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["dimgrey", "teal", "firebrick"]
    for (name, (k_a, c_a)), col in zip(CASES.items(), colors):
        # 81 simulations per case. [:2] to keep (t, x) and drop z
        amp_sim = [steady_measures(*simulate(k_a, c_a, W)[:2], W)[0]
                   for W in Omegas]
        amp_lin = [linear_amplitude(k_a, c_a, W)[0] for W in Omegas]
        ax.plot(Omegas, amp_sim, "-", color=col, label=name)
        ax.plot(Omegas, amp_lin, "--", color=col, lw=0.9)
    # dotted vertical line at the primary natural frequency
    ax.axvline(OMEGA0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"forcing frequency $\Omega$")
    ax.set_ylabel(r"steady-state amplitude max$|x|$")
    ax.set_title(r"Solid: simulation with $\beta=%.1f$. Dashed: exact linear"
                 r" ($\beta = 0$)." % beta)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/q7_frequency_response.pdf")
    plt.close(fig)


# 3. Accuracy checks. See report
def accuracy_checks():
    # Check 1: simulate with beta = 0 and compare with the exact linear
    # amplitude. Catches errors in rhs and solver error.
    print("\nAccuracy check 1: simulation with beta = 0 vs exact linear amplitude")
    for name, (k_a, c_a) in CASES.items():
        for W in (0.8, 1.0, 1.2):
            t, x, _ = simulate(k_a, c_a, W, b=0.0)
            sim = steady_measures(t, x, W)[0]
            exact = linear_amplitude(k_a, c_a, W)[0]
            print(f"  {name:34s} Omega={W:.1f}: sim={sim:.6f} "
                  f"exact={exact:.6f} rel.err={abs(sim - exact) / exact:.1e}")

    # Check 2: no exact solution for beta = 0.2, so rerun with 100x stricter
    # tolerances. If the amplitude doesn't change, solver error is negligible.
    print("\nAccuracy check 2: tolerance tightened 100x (beta = 0.2, Omega = 1)")
    for name, (k_a, c_a) in CASES.items():
        a1 = steady_measures(*simulate(k_a, c_a, 1.0)[:2], 1.0)[0]
        a2 = steady_measures(*simulate(k_a, c_a, 1.0, rtol=1e-11,
                                       atol=1e-14)[:2], 1.0)[0]
        print(f"  {name:34s} rtol 1e-9: {a1:.8f}  rtol 1e-11: {a2:.8f}  "
              f"diff={abs(a1 - a2):.1e}")

    # Check 3: steady state. For beta != 0 a unique steady state is not
    # guaranteed, so compare (a) the amplitude in the second-to-last and last
    # N_MEASURE-period windows (has x stopped changing?), and (b) the last
    # window when starting from a different initial state (same steady state?).
    # Done near the peaks, where problems would show up first.
    print("\nAccuracy check 3: steady state (beta = 0.2)")
    print(f"  {'case':34s} Omega   prev window  last window  other start")
    for name, (k_a, c_a) in CASES.items():
        for W in (0.86, 0.92, 1.0, 1.02):
            t, x, _ = simulate(k_a, c_a, W)
            period = 2 * np.pi / W
            # second-to-last window: between 2*N_MEASURE and N_MEASURE periods before the end
            prev = (t >= T_END - 2 * N_MEASURE * period) & (t < T_END - N_MEASURE * period)
            a_prev = np.max(np.abs(x[prev]))
            a_last = steady_measures(t, x, W)[0]
            # same case and frequency, but starting displaced instead of at rest
            t2, x2, _ = simulate(k_a, c_a, W, y0=(0.6, 0.0, -0.6, 0.0))
            a_other = steady_measures(t2, x2, W)[0]
            print(f"  {name:34s} {W:.2f}  {a_prev:11.6f}  {a_last:11.6f}  {a_other:11.6f}")


# 4. Sensitivity to mistuning?
# Forcing at OMEGA0, absorber tuned to (1 + pct/100) * OMEGA0 for a range of
# small mistunings. Prints max |x| (beta = 0.2) and the ratio to perfect tuning.
def mistuning_sweep():
    print("\nSensitivity to mistuning (forcing at Omega = Omega_0, beta = 0.2)")
    print(f"  {'mistuning':>9s}  {'max|x|':>8s}  {'ratio to tuned':>14s}")
    # perfect tuning as the reference
    k_a, c_a = absorber(OMEGA0)
    a_tuned = steady_measures(*simulate(k_a, c_a, OMEGA0)[:2], OMEGA0)[0]
    for pct in (-10, -5, -2, -1, 0, 1, 2, 5, 10):
        k_a, c_a = absorber((1 + pct / 100) * OMEGA0)
        amp = steady_measures(*simulate(k_a, c_a, OMEGA0)[:2], OMEGA0)[0]
        print(f"  {pct:+8d}%  {amp:8.4f}  {amp / a_tuned:14.2f}")


# MAIN
if __name__ == "__main__":
    # create figures
    os.makedirs("figures", exist_ok=True)
    print("Absorber constants (k_a, c_a):")
    for name, (k_a, c_a) in CASES.items():
        print(f"  {name:34s} k_a={k_a:.4f}  c_a={c_a:.4f}")
    time_series_figure()
    frequency_response_figure()
    accuracy_checks()
    mistuning_sweep()
    print("\nFigures written to ./figures/")
