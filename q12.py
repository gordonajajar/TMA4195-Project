import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.signal import place_poles
import matplotlib.pyplot as plt

#Reused from Q9
from q9 import (M, m, k, c, k_a, c_a, T, beta, P, OM1, OM2,
                RTOL, ATOL, T_END, WINDOW, d, z_d, z_d_dot, u_d,
                steady_amplitude)


#LINEARISED
A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-T**2 * (k + k_a) / M, -T * (c + c_a) / M, T**2 * k_a / M, T * c_a / M],
    [0.0, 0.0, 0.0, 1.0],
    [T**2 * k_a / m, T * c_a / m, -T**2 * k_a / m, -T * c_a / m],
])
B = np.array([0.0, -T**2 / M, 0.0, T**2 / m])
B_d = np.array([0.0, T**2 / M, 0.0, 0.0])

POLES = [-1 + 1j, -1 - 1j, -1.5 + 0.5j, -1.5 - 0.5j]

#d = 0 runs
T_FREE = 20.0
X0_GRID = np.geomspace(0.05, 5.0, 25)
REL_LIMIT = 0.10                


#GAIN

#Pole placement
def gain(poles):
    K = place_poles(A, B.reshape(-1, 1), poles).gain_matrix[0]
    achieved = np.sort_complex(np.linalg.eigvals(A - np.outer(B, K)))
    assert np.allclose(achieved, np.sort_complex(poles)), achieved
    return K


#MODEL

def accelerations(x, v, z, w, u, dist, b):
    a_x = (dist - u - (c / T) * v - (c_a / T) * (v - w)
           - k * x - b * x**3 - k_a * (x - z)) / (M / T**2)
    a_z = (u - (c_a / T) * (w - v) - k_a * (z - x)) / (m / T**2)
    return a_x, a_z


def no_disturbance(t):
    return 0.0


def no_control(t, x, v, z, w):
    return 0.0


def feedback(K):
    k1, k2, k3, k4 = K

    def u(t, x, v, z, w):
        return -(k1 * x + k2 * v + k3 * z + k4 * w)
    return u


def feedforward_feedback(K):
    k1, k2, k3, k4 = K

    def u(t, x, v, z, w):
        return u_d(t) - (k1 * x + k2 * v + k3 * (z - z_d(t))
                         + k4 * (w - z_d_dot(t)))
    return u


#Reduced model (f = u)
def rhs(t, y, u, dist, b):
    x, v, z, w = y
    a_x, a_z = accelerations(x, v, z, w, u(t, x, v, z, w), dist(t), b)
    return [v, a_x, w, a_z]


#Stop a run if it blows up
def blow_up(t, y, *args):
    return 1e3 - np.max(np.abs(y))


blow_up.terminal = True


#SIMULATION FUNCTIONS

# Nonlinear reduced model
def simulate(u, dist, y0, t_end, b=beta, n_out=15001, rtol=RTOL, atol=ATOL):
    t_eval = np.linspace(0.0, t_end, n_out)
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", t_eval=t_eval,
                    rtol=rtol, atol=atol, args=(u, dist, b), events=blow_up)
    if sol.status == 1:
        return None
    assert sol.success, sol.message
    return sol.t, sol.y


#Linearised closed loop with d = 0
def linear_free(K, x0, t):
    E = expm((A - np.outer(B, K)) * (t[1] - t[0]))
    X = np.empty((4, t.size))
    X[:, 0] = [x0, 0.0, 0.0, 0.0]
    for i in range(1, t.size):
        X[:, i] = E @ X[:, i - 1]
    return X

def relative_difference(K, x0, n_out=4001):
    t = np.linspace(0.0, T_FREE, n_out)
    x_lin = linear_free(K, x0, t)[0]
    res = simulate(feedback(K), no_disturbance, [x0, 0.0, 0.0, 0.0], T_FREE,
                   n_out=n_out)
    if res is None:
        return np.inf
    return np.max(np.abs(res[1][0] - x_lin)) / np.max(np.abs(x_lin))


#Smallest x0 where the relative difference reaches REL_LIMIT (bisection
def threshold(K, grid, values):
    above = np.nonzero(np.asarray(values) >= REL_LIMIT)[0]
    if above.size == 0:
        return None
    lo, hi = grid[above[0] - 1], grid[above[0]]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if relative_difference(K, mid) >= REL_LIMIT:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def exact_linear_amplitude(A_cl):
    t = np.linspace(T_END - WINDOW, T_END, 20001)
    x = np.zeros_like(t)
    for amp, om in [(P, OM1), (0.5 * P, OM2)]:
        G = np.linalg.solve(1j * om * np.eye(4) - A_cl, B_d)[0]
        x += np.real(G * amp * np.exp(1j * om * t))
    return np.max(np.abs(x))


#FIGS

def nonlinear_figure(K):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5))
    t = np.linspace(0.0, T_FREE, 4001)
    axes[0].plot(t, linear_free(K, 1.0, t)[0], "k--", lw=1.0,
                 label="linearised (any $x_0$)")
    for x0 in (0.5, 1.5, 3.0):
        _, y = simulate(feedback(K), no_disturbance, [x0, 0.0, 0.0, 0.0],
                        T_FREE, n_out=4001)
        axes[0].plot(t, y[0] / x0, label=f"nonlinear, $x_0 = {x0}$")
    axes[0].set_xlim(0, 10)
    axes[0].set_xlabel("rescaled time $t$")
    axes[0].set_ylabel("$x(t)/x_0$")
    axes[0].legend()

    print(f"\nLinearised against nonlinear (d = 0, beta = {beta}):")
    print(f"  {'x0':>6s}  {'rel.diff':>10s}  {'rel.diff/x0^2':>13s}")
    rel = [relative_difference(K, x0) for x0 in X0_GRID]
    for x0, r in zip(X0_GRID, rel):
        print(f"  {x0:6.3f}  {r:10.3e}  {r / x0**2:13.4f}")
    axes[1].loglog(X0_GRID, rel, "o-", ms=3, color="firebrick",
                   label="nonlinear against linearised")
    axes[1].axhline(REL_LIMIT, color="grey", lw=0.8, ls="--",
                    label=f"{REL_LIMIT:.0%} difference")
    axes[1].set_xlabel("initial displacement $x_0$")
    axes[1].set_ylabel(r"$\max|x_{\mathrm{nl}} - x_{\mathrm{lin}}|"
                       r" / \max|x_{\mathrm{lin}}|$")
    axes[1].legend()
    axes[1].grid(which="minor", color="0.9")
    fig.tight_layout()
    fig.savefig("figures/q12_nonlinear.pdf")
    plt.close(fig)

    x_star = threshold(K, X0_GRID, rel)
    print(f"  relative difference reaches {REL_LIMIT:.0%} at x0 = {x_star:.3f}"
          f"  (sqrt(k/beta) = {np.sqrt(k / beta):.3f})")
    x0 = X0_GRID[-1]
    t, y = simulate(feedback(K), no_disturbance, [x0, 0.0, 0.0, 0.0], T_FREE)
    print(f"  x0 = {x0}: |X(t)| at t = {T_FREE:.0f} is {np.linalg.norm(y[:, -1]):.1e}")


#second one
def disturbed_figure(K):
    y0 = [0.0, 0.0, 0.0, 0.0]
    runs = [("no control ($u = 0$)", no_control),
            ("(iii) $u = -KX$", feedback(K)),
            ("(iv) $u = u_d - K(X - X_d)$", feedforward_feedback(K))]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    print("\nDisturbance from Q9, X(0) = 0, nonlinear reduced model, "
          "steady state (last 20 pi):")
    print(f"  {'case':<28s}  {'max|x|':>9s}  {'max|z|':>9s}  {'max|z-z_d|':>10s}")
    for label, u in runs:
        t, y = simulate(u, d, y0, T_END)
        window = t >= T_END - WINDOW
        print(f"  {label:<28s}  {steady_amplitude(t, y[0]):9.3e}  "
              f"{np.max(np.abs(y[2][window])):9.3e}  "
              f"{np.max(np.abs(y[2] - z_d(t))[window]):10.3e}")
        axes[0].plot(t, y[0], label=label)
        axes[1].plot(t, y[2])
        if label.startswith("(iv)"):
            late = t[np.abs(y[0]) >= 1e-3]
            print(f"  (iv): |x| < 1e-3 for all t >= {late[-1]:.2f}, "
                  f"max|x| = {np.max(np.abs(y[0])):.3e} at "
                  f"t = {t[np.argmax(np.abs(y[0]))]:.2f}")
    axes[1].plot(t, z_d(t), "k:", label="desired $z_d$")
    axes[0].set_ylabel("$x(t)$")
    axes[1].set_ylabel("$z(t)$")
    axes[1].set_xlabel("rescaled time $t$")
    axes[0].set_ylim(-0.13, 0.2)
    axes[0].legend(loc="upper center", ncol=3, fontsize=8)
    axes[1].set_ylim(-0.13, 0.17)
    axes[1].legend(loc="upper center", fontsize=8)
    axes[0].set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig("figures/q12_disturbed.pdf")
    plt.close(fig)


#accuracy check
def accuracy_checks(K):
    x0 = 0.5
    t = np.linspace(0.0, T_FREE, 4001)
    x_exact = linear_free(K, x0, t)[0]
    _, y = simulate(feedback(K), no_disturbance, [x0, 0.0, 0.0, 0.0], T_FREE,
                    b=0.0, n_out=4001)
    print("\nAccuracy check 1: beta = 0, d = 0, x0 = 0.5, against exp((A - BK)t) X(0)")
    print(f"  max|x_sim - x_exact| = {np.max(np.abs(y[0] - x_exact)):.1e}")
    print("\nAccuracy check 2: beta = 0, Q9 disturbance, steady max|x|")
    for label, u, A_cl in [("no control", no_control, A),
                           ("(iii) u = -KX", feedback(K), A - np.outer(B, K))]:
        t, y = simulate(u, d, [0.0, 0.0, 0.0, 0.0], T_END, b=0.0)
        sim, exact = steady_amplitude(t, y[0]), exact_linear_amplitude(A_cl)
        print(f"  {label:<14s}: simulated {sim:.6e}, exact {exact:.6e}, "
              f"rel.diff {abs(sim - exact) / exact:.1e}")


#MAIN
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("open-loop eigenvalues of A:",
          np.round(np.sort_complex(np.linalg.eigvals(A)), 4))
    K = gain(POLES)
    print(f"poles {POLES}")
    print(f"K = {np.array2string(K, precision=4)}")
    print("A - BK =\n", np.round(A - np.outer(B, K), 6))
    print("characteristic polynomial of A - BK:",
          np.round(np.real(np.poly(A - np.outer(B, K))), 6))
    nonlinear_figure(K)
    disturbed_figure(K)
    accuracy_checks(K)
    print("\nFigures written to ./figures/")
