import sympy as sp

M, m, k, c, ka, ca, beta, T = sp.symbols("M m k c k_a c_a beta T", positive=True)
x, v, z, w, u, d = sp.symbols("x v z w u d")

#Reduced model
F = sp.Matrix([
    v,
    T**2 / M * (d - u - c / T * v - ca / T * (v - w) - k * x - beta * x**3 - ka * (x - z)),
    w,
    T**2 / m * (u - ca / T * (w - v) - ka * (z - x)),
])
X = sp.Matrix([x, v, z, w])
origin = {x: 0, v: 0, z: 0, w: 0, u: 0, d: 0}

assert F.subs(origin) == sp.zeros(4, 1), "origin is not an equilibrium"

A = sp.simplify(F.jacobian(X).subs(origin))
B = sp.simplify(F.jacobian(sp.Matrix([u])).subs(origin))

base = {M: 1, m: 1, k: 1, c: 1, T: 1}
A7, B7 = A.subs(base), B.subs(base)
I4 = sp.eye(4)

# All sensor configurations
sensors = {
    "y = x":         (I4[0, :], sp.zeros(1, 1)),
    "y = z":         (I4[2, :], sp.zeros(1, 1)),
    "y = xdd":       (A7[1, :], B7[1, :]),
    "y = zdd":       (A7[3, :], B7[3, :]),
    "y = (xdd,zdd)": (A7.extract([1, 3], [0, 1, 2, 3]), B7.extract([1, 3], [0])),
}

def observability(A, C):
    return sp.Matrix.vstack(C, C * A, C * A**2, C * A**3)

print("C, D and symbolic det(O) for each sensor:")
for name, (C, D) in sensors.items():
    print(f"\n{name}:  C = {list(C)}   D = {list(D)}")
    O = observability(A7, C)
    if O.is_square:
        print("  det O =", sp.factor(O.det()))

cases = [{ka: 1, ca: 1}, {ka: 1, ca: 0}, {ka: 0, ca: 1}]

print("\nRank table (observable iff rank = 4):")
print(" " * 16 + "".join(f"{name:>16}" for name in sensors))
for case in cases:
    row = f"ka={case[ka]}, ca={case[ca]}".ljust(16)
    for name, (C, D) in sensors.items():
        r = observability(A7.subs(case), C.subs(case)).rank()
        row += f"{('obs' if r == 4 else 'NOT obs') + f' (r={r})':>16}".rjust(16)
    print(row)



