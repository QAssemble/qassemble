import logging
import os

import h5py
import numpy as np

logger = logging.getLogger("QAssemble")


class IO:
    """Small HDF5 IO helpers used by QAssemble objects."""

    @staticmethod
    def Group(handle, *path):
        """Return a nested HDF5 group, creating missing groups."""
        if handle is None:
            raise ValueError("HDF5 handle is required")
        if not path:
            raise ValueError("HDF5 group path is required")

        group = handle
        for name in path:
            if name is None:
                raise ValueError("HDF5 group path cannot contain None")
            group = group.require_group(str(name))
        return group

    @staticmethod
    def CreateDataset(group, name: str, data, dtype=None):
        """Create or replace a dataset in an HDF5 group."""
        if name in group:
            del group[name]
        if dtype is None:
            return group.create_dataset(name, data=data)
        return group.create_dataset(name, data=data, dtype=dtype)

    @staticmethod
    def MixComponent(
        hdf5file: str,
        group: str,
        key,
        component: str,
        value,
        iter: int,
        mix: float,
        method: str,
        npulay: int,
        mixer,
    ) -> np.ndarray:
        """Mix one component using HDF5 as the source of persisted history."""
        if hdf5file is None:
            raise ValueError("MixComponent requires hdf5file")
        if group is None:
            raise ValueError("MixComponent requires group")
        if key is None:
            raise ValueError("MixComponent requires key")
        if component is None:
            raise ValueError("MixComponent requires component")
        if mixer is None:
            raise ValueError("MixComponent requires mixer")
        if iter is None:
            raise ValueError("MixComponent requires iter")
        if int(iter) <= 0:
            raise ValueError("iter must be positive")
        if mix is None:
            raise ValueError("MixComponent requires mix")
        if not (0.0 < float(mix) <= 1.0):
            raise ValueError("mix must satisfy 0 < mix <= 1")

        method = str(method).lower()
        if method not in {"linear", "pulay"}:
            raise ValueError(f"Unknown mixing method: {method}")

        npulay = int(npulay)
        if npulay < 1:
            raise ValueError("npulay must be positive")

        fnew = IO._as_mixing_array(component, value)
        shape = tuple(fnew.shape)

        # After the first iteration the file must already hold the mixing
        # history.  Mode "a" would otherwise create an empty one and every call
        # would silently fall through to the reset branch below, returning the
        # input unmixed -- which is exactly how a relative hdf5file combined
        # with a per-iteration os.chdir used to disable mixing entirely.
        if int(iter) > 1 and not os.path.exists(hdf5file):
            raise ValueError(
                f"MixComponent: hdf5file does not exist at iter={int(iter)}: "
                f"{hdf5file!r} (cwd={os.getcwd()!r}). The mixing history would "
                f"be lost; pass an absolute path."
            )

        with h5py.File(hdf5file, "a") as handle:
            comp_group = IO.Group(handle, group, "Mixing", key, component)
            comp_group.attrs["method"] = method
            comp_group.attrs["npulay"] = npulay

            if int(iter) == 1 or "last" not in comp_group:
                if int(iter) != 1:
                    logger.warning(
                        f"[mixing] {key}/{component} iter={int(iter)}: no stored "
                        f"'last' in {hdf5file!r}; resetting history and passing "
                        f"the input through UNMIXED"
                    )
                else:
                    logger.info(
                        f"[mixing] {key}/{component} iter=1: history reset "
                        f"(first iteration is never mixed)"
                    )
                IO._reset_mixing_component(comp_group)
                IO._write_mixing_dataset(comp_group, "last", fnew)
                IO._write_mixing_attrs(
                    comp_group,
                    shape=shape,
                    iter=int(iter),
                    num_history=0,
                    next_slot=0,
                )
                return fnew.copy(order="F")

            stored_shape = tuple(int(x) for x in comp_group.attrs.get("shape", []))
            if stored_shape and stored_shape != shape:
                raise ValueError(
                    f"{component} shape mismatch: HDF5 has {stored_shape}, got {shape}"
                )

            fold = np.asarray(comp_group["last"][()], dtype=np.complex128)
            if fold.shape != shape:
                raise ValueError(
                    f"{component} last shape mismatch: expected {shape}, got {fold.shape}"
                )

            if method == "linear":
                mixed = mixer(method=method, mix=float(mix), fnew=fnew, fold=fold)
                IO._write_mixing_dataset(comp_group, "last", mixed)
                IO._write_mixing_attrs(comp_group, shape=shape, iter=int(iter))
                logger.info(
                    f"[mixing] {key}/{component} iter={int(iter)}: linear "
                    f"mix={float(mix)} applied"
                )
                return mixed

            residual = np.asfortranarray(fnew - fold)
            IO._append_mixing_history(
                comp_group,
                fold=fold,
                residual=residual,
                npulay=npulay,
            )
            inputs, residuals = IO._read_mixing_history(comp_group, npulay=npulay)
            mixed = mixer(
                method=method,
                mix=float(mix),
                fnew=fnew,
                fold=fold,
                inputs=inputs,
                residuals=residuals,
            )
            IO._write_mixing_dataset(comp_group, "last", mixed)
            IO._write_mixing_attrs(comp_group, shape=shape, iter=int(iter))
            logger.info(
                f"[mixing] {key}/{component} iter={int(iter)}: pulay "
                f"mix={float(mix)} applied with num_history="
                f"{int(comp_group.attrs.get('num_history', 0))}/{npulay}"
            )
            return mixed

    @staticmethod
    def _as_mixing_array(component: str, value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.complex128)
        if arr.size == 0:
            raise ValueError(f"{component} must be non-empty")
        if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
            raise ValueError(f"{component} contains non-finite values")
        return np.asfortranarray(arr)

    @staticmethod
    def _write_mixing_dataset(group, name: str, data) -> None:
        IO.CreateDataset(group, name, np.asarray(data, dtype=np.complex128))

    @staticmethod
    def _write_mixing_attrs(
        comp_group,
        shape: tuple,
        iter: int,
        num_history: int = None,
        next_slot: int = None,
    ) -> None:
        comp_group.attrs["shape"] = np.asarray(shape, dtype=np.int64)
        comp_group.attrs["last_iter"] = int(iter)
        if num_history is not None:
            comp_group.attrs["num_history"] = int(num_history)
        if next_slot is not None:
            comp_group.attrs["next_slot"] = int(next_slot)

    @staticmethod
    def _reset_mixing_component(comp_group) -> None:
        for name in ("last", "input_history", "residual_history"):
            if name in comp_group:
                del comp_group[name]
        comp_group.require_group("input_history")
        comp_group.require_group("residual_history")

    @staticmethod
    def _append_mixing_history(
        comp_group,
        fold: np.ndarray,
        residual: np.ndarray,
        npulay: int,
    ) -> None:
        input_group = comp_group.require_group("input_history")
        residual_group = comp_group.require_group("residual_history")

        num_history = int(comp_group.attrs.get("num_history", 0))
        next_slot = int(comp_group.attrs.get("next_slot", 0))
        slot = next_slot % int(npulay)
        slot_name = str(slot)

        IO._write_mixing_dataset(input_group, slot_name, fold)
        IO._write_mixing_dataset(residual_group, slot_name, residual)

        comp_group.attrs["num_history"] = min(num_history + 1, int(npulay))
        comp_group.attrs["next_slot"] = (slot + 1) % int(npulay)

    @staticmethod
    def _read_mixing_history(comp_group, npulay: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        num_history = int(comp_group.attrs.get("num_history", 0))
        next_slot = int(comp_group.attrs.get("next_slot", 0))
        if num_history <= 0:
            return [], []

        npulay = int(npulay)
        if num_history < npulay:
            slots = list(range(num_history))
        else:
            slots = list(range(next_slot, npulay)) + list(range(0, next_slot))

        input_group = comp_group["input_history"]
        residual_group = comp_group["residual_history"]
        inputs = []
        residuals = []
        for slot in slots:
            slot_name = str(slot)
            if slot_name not in input_group or slot_name not in residual_group:
                continue
            inputs.append(np.asarray(input_group[slot_name][()], dtype=np.complex128))
            residuals.append(np.asarray(residual_group[slot_name][()], dtype=np.complex128))

        return inputs, residuals
