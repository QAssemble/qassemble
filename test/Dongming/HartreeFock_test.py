import numpy as np
import matplotlib.pyplot as plt

# ---------- PARAMETERS ----------
# Define symbolic parameters (to be filled later)
params = {
    't': 1.0,             # hopping
    'U': 5.0,             # on-site intra-orbital repulsion
    'U_prime': 2.0,       # on-site inter-orbital repulsion
    'J_onsite': 0.5,      # on-site exchange
    'V': 1.0,             # inter-site Coulomb
    'J_intersite': 0.1,   # inter-site exchange
}

# Spin factor: spin degeneracy is included by default (spinful system)
n_half = 1.0 # 0.5  # half occupancy per spin

tol = 1e-5
max_iter = 100

# ---------- LATTICE SETUP ----------
# Real-space lattice vectors (3D)
a1 = np.array([1.0, 0.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2, 0.0])
a3 = np.array([0.0, 0.0, 10.0])
lattice_vectors = np.array([a1, a2, a3])

def reciprocal_vector(a1, a2, a3):
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    return b1, b2, b3

b1, b2, b3 = reciprocal_vector(a1, a2, a3)

# Basis vectors (fractional)
tau_A = np.array([1/3, 1/3, 0.0])
tau_B = np.array([2/3, 2/3, 0.0])
delta_frac = np.array([
    tau_B - tau_A,
    tau_B - tau_A - [1, 0, 0],
    tau_B - tau_A - [0, 1, 0]
])
delta_cart = delta_frac @ lattice_vectors

# ---------- K-PATH SETUP ----------
# High-symmetry points
Gamma = np.array([0.0, 0.0, 0.0])
K = (2*b1 + b2)/3
M = 0.5 * b1
K = np.array([4*np.pi/3/1,0,0])

k_path = [Gamma, K, M, Gamma]
k_labels = [r'$\Gamma$', r'$K$', r'$M$', r'$\Gamma$']
num_k = 100
k_points = []
for i in range(len(k_path) - 1):
    for j in range(num_k):
        k = k_path[i] + (k_path[i+1] - k_path[i]) * j / num_k
        k_points.append(k)
k_points = np.array(k_points)

# ---------- BOND ORDER PARAMETERS (initialize constant) ----------
chi_up = 0.1
chi_dn = 0.1








# ---------- BANDSTRUCTURE CALCULATION ----------
# bands = []

# for k in k_points:
#     # Structure factor f(k)
#     f_k = np.sum(np.exp(1j * np.dot(delta_cart, k)))

#     # Initialize 4x4 Hamiltonian
#     H = np.zeros((4, 4), dtype=complex)

#     # Fill kinetic term
#     H[0, 2] = -params['t'] * f_k
#     H[1, 3] = -params['t'] * f_k
#     H[2, 0] = -params['t'] * np.conj(f_k)
#     H[3, 1] = -params['t'] * np.conj(f_k)

#     # On-site Slater-Kanamori U, U'
#     H[0, 0] += params['U'] * n_half + params['U_prime'] * (n_half + 2 * n_half)
#     H[1, 1] += params['U'] * n_half + params['U_prime'] * (n_half + 2 * n_half)
#     H[2, 2] += params['U'] * n_half + params['U_prime'] * (n_half + 2 * n_half)
#     H[3, 3] += params['U'] * n_half + params['U_prime'] * (n_half + 2 * n_half)

#     # On-site exchange J_onsite (zero in single-orbital case)
#     # Skipped as no exchange between orbitals

#     # Inter-site Coulomb V
#     n_A = 2 * n_half
#     n_B = 2 * n_half
#     H[0, 0] += params['V'] * n_B
#     H[1, 1] += params['V'] * n_B
#     H[2, 2] += params['V'] * n_A
#     H[3, 3] += params['V'] * n_A

#     # Inter-site exchange J_intersite
#     H[0, 2] += -params['J_intersite'] * chi_up
#     H[1, 3] += -params['J_intersite'] * chi_dn
#     H[2, 0] += -params['J_intersite'] * chi_up
#     H[3, 1] += -params['J_intersite'] * chi_dn

#     # Diagonalize
#     eigvals = np.linalg.eigvalsh(H)
#     bands.append(eigvals.real)












# ---------- SELF-CONSISTENT FIELD CALCULATION ----------
n_A_up, n_A_dn = n_half, n_half
n_B_up, n_B_dn = n_half, n_half
chi_up = 0.1
chi_dn = 0.1

for iteration in range(max_iter):
    n_A_up_new = n_A_dn_new = 0.0
    n_B_up_new = n_B_dn_new = 0.0
    chi_up_new = 0.0
    chi_dn_new = 0.0
    bands = []

    for k in k_points:
        # Structure factor f(k)
        f_k = np.sum(np.exp(1j * np.dot(delta_cart, k)))

        H = np.zeros((4, 4), dtype=complex)

        # Kinetic term
        H[0, 2] = -params['t'] * f_k
        H[1, 3] = -params['t'] * f_k
        H[2, 0] = -params['t'] * np.conj(f_k)
        H[3, 1] = -params['t'] * np.conj(f_k)

        # On-site Slater-Kanamori
        H[0, 0] += params['U'] * n_A_dn + params['U_prime'] * (n_A_dn + n_B_up + n_B_dn)
        H[1, 1] += params['U'] * n_A_up + params['U_prime'] * (n_A_up + n_B_up + n_B_dn)
        H[2, 2] += params['U'] * n_B_dn + params['U_prime'] * (n_B_dn + n_A_up + n_A_dn)
        H[3, 3] += params['U'] * n_B_up + params['U_prime'] * (n_B_up + n_A_up + n_A_dn)

        # Inter-site Coulomb
        n_A = n_A_up + n_A_dn
        n_B = n_B_up + n_B_dn
        H[0, 0] += params['V'] * n_B
        H[1, 1] += params['V'] * n_B
        H[2, 2] += params['V'] * n_A
        H[3, 3] += params['V'] * n_A

        # Inter-site exchange
        H[0, 2] += -params['J_intersite'] * chi_up
        H[1, 3] += -params['J_intersite'] * chi_dn
        H[2, 0] += -params['J_intersite'] * chi_up
        H[3, 1] += -params['J_intersite'] * chi_dn

        # Diagonalize
        eigvals, eigvecs = np.linalg.eigh(H)
        bands.append(eigvals.real)

        for n in range(4):  # Assume 2 lowest states are occupied
            occ = 1.0 if n < 2 else 0.0
            psi = eigvecs[:, n]
            n_A_up_new += occ * abs(psi[0])**2 / len(k_points)
            n_A_dn_new += occ * abs(psi[1])**2 / len(k_points)
            n_B_up_new += occ * abs(psi[2])**2 / len(k_points)
            n_B_dn_new += occ * abs(psi[3])**2 / len(k_points)
            chi_up_new += occ * np.conj(psi[0]) * psi[2] / len(k_points)
            chi_dn_new += occ * np.conj(psi[1]) * psi[3] / len(k_points)

    # Convergence check
    delta = max(
        abs(n_A_up_new - n_A_up),
        abs(n_A_dn_new - n_A_dn),
        abs(n_B_up_new - n_B_up),
        abs(n_B_dn_new - n_B_dn),
        abs(chi_up_new - chi_up),
        abs(chi_dn_new - chi_dn)
    )

    n_A_up, n_A_dn = n_A_up_new, n_A_dn_new
    n_B_up, n_B_dn = n_B_up_new, n_B_dn_new
    chi_up, chi_dn = chi_up_new, chi_dn_new

    if delta < tol:
        break


print('iteration --', iteration)

bands = np.array(bands)

# ---------- PLOTTING ----------
plt.figure(figsize=(8, 6))
for i in range(4):
    plt.plot(bands[:, i], label=f'Band {i+1}')
plt.xticks([0, num_k, 2*num_k, 3*num_k], k_labels)
plt.xlabel("k-path")
plt.ylabel("Energy")
plt.title("Bandstructure with Slater-Kanamori and Inter-site Interactions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
