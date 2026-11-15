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
Bd = sp.simplify(F.jacobian(sp.Matrix([d])).subs(origin))
print("Question 10: general parameters")
sp.pprint(A)
print("B   =", B.T)
print("B_d =", Bd.T)

ones = {M: 1, m: 1, k: 1, c: 1, ka: 1, ca: 1, T: 1}
A1, B1 = A.subs(ones), B.subs(ones)
print("\nQuestion 10: all parameters = 1")
sp.pprint(A1)
print("B =", B1.T)

lam = sp.symbols("lambda")
charpoly = sp.expand((lam * sp.eye(4) - A1).det())
print("\ndet(lambda I - A) =", charpoly)
print("factored:", sp.factor(charpoly, extension=sp.sqrt(5)))
print("eigenvalues:", [complex(sp.N(e)) for e in sp.roots(charpoly, lam)])


def kalman(A, B):
    return sp.Matrix.hstack(B, A * B, A**2 * B, A**3 * B)


C1 = kalman(A1, B1)
print("\nQuestion 11: Kalman matrix, all parameters = 1")
sp.pprint(C1)
print("det =", C1.det(), "  rank =", C1.rank())

Cg = kalman(A, B)
print("\nQuestion 11: det for general parameters =", sp.factor(Cg.det()))
for label, sub in [("k = 0", {k: 0}), ("k_a = c_a = 0", {ka: 0, ca: 0})]:
    print(f"  rank with {label}: {Cg.subs(sub).rank()}")

Pi = M / T**2 * v + m / T**2 * w + c / T * x
Pi_dot = sp.expand((sp.Matrix([Pi]).jacobian(X) * (A * X + B * u + Bd * d))[0])
print("\nd/dt Pi (linearised) =", sp.simplify(Pi_dot))
