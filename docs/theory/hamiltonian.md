# Hamiltonian

QAssemble uses a Python dict-based input to define the Hamiltonian, split into a one-body (`OneBody`) and a two-body (`TwoBody`) part.

```python
Hamiltonian = {
    'OneBody': {
        'Hopping': {
            ((0,0),(1,0)): {
                1.0: [[0,0,0],[-1,0,0],[0,-1,0]],
            },
        },
        'Onsite': {
            0: {(0,0): 0.0, (1,0): 0.0}
        }
    },
    'TwoBody': {
        'Local': {
            'Parameter': 'SlaterKanamori',
            'option': {
                (0,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0},
                (1,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0}
            }
        },
        'NonLocal': {
            ((0,0),(1,0)): {
                0.20: [[0,0,0],[-1,0,0],[0,-1,0]],
            },
        }
    }
}
```

---

## OneBody

The one-body part describes the non-interacting Hamiltonian $H_0$:

$$H_0 = \sum_{ij} t_{ij} c_i^\dagger c_j + \sum_i \epsilon_i c_i^\dagger c_i$$

### Hopping

Defines the hopping amplitudes $t_{ij}$ between orbital pairs.

```python
'Hopping': {
    ((0,0),(1,0)): {        # orbital pair: (sublattice 0, orbital 0) → (sublattice 1, orbital 0)
        1.0: [[0,0,0],[-1,0,0],[0,-1,0]],   # amplitude: list of lattice vectors (R)
    },
},
```

- **Key** `((i, orb_i), (j, orb_j))` — source and target (sublattice, orbital) pair
- **Sub-key** — hopping amplitude $t$
- **Value** — list of lattice vectors $\mathbf{R}$ connecting the two sites

### Onsite

Defines on-site energies $\epsilon_i$ for each orbital.

```python
'Onsite': {
    0: {(0,0): 0.0, (1,0): 0.0}   # spin index: {(sublattice, orbital): energy}
},
```

---

## TwoBody

The two-body part defines Coulomb interaction terms.

### Local

Specifies the on-site (local) interaction using a predefined parameterization.

```python
'Local': {
    'Parameter': 'SlaterKanamori',   # interaction type
    'option': {
        (0,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0},   # (sublattice, (orbital,)): params
        (1,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0}
    }
},
```

Supported parameterizations:

| Parameter | Description |
|---|---|
| `SlaterKanamori` | Slater-Kanamori interaction ($U$, $U'$, $J$) |
| `Slater` | Full Slater integrals |
| `Kanamori` | Simplified Kanamori form |

### NonLocal

Defines inter-site (non-local) density-density interactions.

```python
'NonLocal': {
    ((0,0),(1,0)): {          # orbital pair
        0.20: [[0,0,0],[-1,0,0],[0,-1,0]],   # amplitude: list of lattice vectors
    },
},
```

Supported non-local interaction types:

| Type | Description |
|---|---|
| `Ohno` | Ohno interpolation $V(r) = U / \sqrt{1 + (r/a)^2}$ |
| `Ohno-Yukawa` | Screened Ohno form |
| `JTH` | J-threading (JTH) |
