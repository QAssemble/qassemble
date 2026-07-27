# Migrating to QAssemble 0.2

QAssemble 0.2 uses the class names from the manuscript as its only supported
public API. Update Python code using the following mapping.

| QAssemble 0.1 | QAssemble 0.2 |
|---|---|
| `NIHamiltonian` | `H0` |
| `Hamiltonian` | `H` |
| `SigmaHartree` | `SigH` |
| `SigmaFock` | `SigF` |
| `GreenBare` | `G0` |
| `GreenInt` | `G` |
| `SigmaGWC` | `SigGWC` |
| `PolLat` | `P` |
| `WLat` | `W` |
| `VBare` | `V` |

## Migrating HDF5 results

Preview the groups that will be renamed:

```bash
qassemble-migrate-hdf5 --dry-run result.h5
```

Then migrate the file in place:

```bash
qassemble-migrate-hdf5 result.h5
```

The command validates the complete operation first, creates
`result.h5.pre-class-rename.bak`, performs the migration in a temporary copy,
and atomically replaces the original after verification. Use `--backup PATH`
to select a different backup path. The `/input/Hamiltonian` configuration group
is preserved because it is an input section rather than a class name.

After migration, restart and spectral post-processing expect only the new HDF5
group names.
