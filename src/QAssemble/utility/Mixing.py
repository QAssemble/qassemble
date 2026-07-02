import h5py
import numpy as np


class Mixing:
    """HDF5-backed multi-component mixing for SCF iterations.

    The mixer is intentionally stateless: both linear mixing and Pulay/DIIS
    use only the state stored in HDF5.  Callers must pass a dictionary of
    component arrays through ``quantities``.
    """

    def __init__(
        self,
        method: str = "pulay",
        npulay: int = 5,
        hdf5file: str = None,
        group: str = None,
    ):
        self.method = method.lower()
        if self.method not in {"linear", "pulay"}:
            raise ValueError(f"Unknown mixing method: {method}")
        self.npulay = int(npulay)
        if self.npulay < 1:
            raise ValueError("npulay must be positive")
        if hdf5file is None:
            raise ValueError("Mixing requires hdf5file")
        if group is None:
            raise ValueError("Mixing requires group")
        self.hdf5file = hdf5file
        self.group = group

    def __call__(
        self,
        iter: int = None,
        mix: float = None,
        key=None,
        quantities: dict = None,
        **kwargs,
    ) -> dict:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(
                "Mixing only supports multi-component quantities; "
                f"unsupported arguments: {unsupported}"
            )
        if iter is None:
            raise ValueError("Mixing requires iter")
        if mix is None:
            raise ValueError("Mixing requires mix")
        if key is None:
            raise ValueError("Mixing requires key")
        if quantities is None:
            raise ValueError("Mixing requires quantities")
        if not isinstance(quantities, dict) or len(quantities) == 0:
            raise ValueError("quantities must be a non-empty dict")
        if not (0.0 < float(mix) <= 1.0):
            raise ValueError("mix must satisfy 0 < mix <= 1")

        mixed = {}
        with h5py.File(self.hdf5file, "a") as handle:
            root = self._root_group(handle)
            for component, value in quantities.items():
                arr = self._as_array(component, value)
                comp_group = self._component_group(root, key, component)
                mixed[component] = self._mix_component(
                    comp_group=comp_group,
                    component=component,
                    iter=int(iter),
                    mix=float(mix),
                    fnew=arr,
                )

        return mixed

    def _root_group(self, handle):
        group = handle.require_group(str(self.group))
        return group.require_group("Mixing")

    def _component_group(self, root, key, component: str):
        key_group = root.require_group(str(key))
        comp_group = key_group.require_group(str(component))
        comp_group.attrs["method"] = self.method
        comp_group.attrs["npulay"] = self.npulay
        return comp_group

    @staticmethod
    def _as_array(component: str, value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.complex128)
        if arr.size == 0:
            raise ValueError(f"{component} must be non-empty")
        if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
            raise ValueError(f"{component} contains non-finite values")
        return np.asfortranarray(arr)

    def _mix_component(
        self,
        comp_group,
        component: str,
        iter: int,
        mix: float,
        fnew: np.ndarray,
    ) -> np.ndarray:
        if iter <= 0:
            raise ValueError("iter must be positive")

        shape = tuple(fnew.shape)
        has_last = "last" in comp_group
        if iter == 1 or not has_last:
            self._reset_component(comp_group)
            self._write_dataset(comp_group, "last", fnew)
            self._write_attrs(comp_group, shape=shape, iter=iter, num_history=0, next_slot=0)
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

        if self.method == "linear":
            mixed = self._linear(mix=mix, fnew=fnew, fold=fold)
            self._write_dataset(comp_group, "last", mixed)
            self._write_attrs(comp_group, shape=shape, iter=iter)
            return mixed

        return self._pulay(comp_group=comp_group, shape=shape, iter=iter, mix=mix, fnew=fnew, fold=fold)

    @staticmethod
    def _linear(mix: float, fnew: np.ndarray, fold: np.ndarray) -> np.ndarray:
        return np.asfortranarray(mix * fnew + (1.0 - mix) * fold)

    def _pulay(
        self,
        comp_group,
        shape: tuple,
        iter: int,
        mix: float,
        fnew: np.ndarray,
        fold: np.ndarray,
    ) -> np.ndarray:
        residual = np.asfortranarray(fnew - fold)
        self._append_history(comp_group, fold=fold, residual=residual)

        inputs, residuals = self._read_history(comp_group)
        if len(inputs) < 2:
            mixed = self._linear(mix=mix, fnew=fnew, fold=fold)
            self._write_dataset(comp_group, "last", mixed)
            self._write_attrs(comp_group, shape=shape, iter=iter)
            return mixed

        nhist = len(inputs)
        overlap = np.zeros((nhist, nhist), dtype=np.float64)
        residual_vecs = [res.ravel() for res in residuals]
        for i in range(nhist):
            for j in range(i, nhist):
                value = np.vdot(residual_vecs[i], residual_vecs[j]).real
                overlap[i, j] = value
                overlap[j, i] = value

        try:
            coeff = np.linalg.inv(overlap).sum(axis=1)
        except np.linalg.LinAlgError:
            mixed = self._linear(mix=mix, fnew=fnew, fold=fold)
            self._write_dataset(comp_group, "last", mixed)
            self._write_attrs(comp_group, shape=shape, iter=iter)
            return mixed

        coeff_sum = coeff.sum()
        if abs(coeff_sum) < 1.0e-14:
            mixed = self._linear(mix=mix, fnew=fnew, fold=fold)
            self._write_dataset(comp_group, "last", mixed)
            self._write_attrs(comp_group, shape=shape, iter=iter)
            return mixed
        coeff /= coeff_sum

        mixed_flat = np.zeros(inputs[0].size, dtype=np.complex128)
        for ci, input_i, residual_i in zip(coeff, inputs, residuals):
            mixed_flat += ci * (input_i.ravel() + mix * residual_i.ravel())
        mixed = np.asfortranarray(mixed_flat.reshape(shape))
        self._write_dataset(comp_group, "last", mixed)
        self._write_attrs(comp_group, shape=shape, iter=iter)
        return mixed

    def _append_history(self, comp_group, fold: np.ndarray, residual: np.ndarray) -> None:
        input_group = comp_group.require_group("input_history")
        residual_group = comp_group.require_group("residual_history")

        num_history = int(comp_group.attrs.get("num_history", 0))
        next_slot = int(comp_group.attrs.get("next_slot", 0))
        slot = next_slot % self.npulay
        slot_name = str(slot)

        self._write_dataset(input_group, slot_name, fold)
        self._write_dataset(residual_group, slot_name, residual)

        comp_group.attrs["num_history"] = min(num_history + 1, self.npulay)
        comp_group.attrs["next_slot"] = (slot + 1) % self.npulay

    def _read_history(self, comp_group) -> tuple[list[np.ndarray], list[np.ndarray]]:
        num_history = int(comp_group.attrs.get("num_history", 0))
        next_slot = int(comp_group.attrs.get("next_slot", 0))
        if num_history <= 0:
            return [], []

        if num_history < self.npulay:
            slots = list(range(num_history))
        else:
            slots = list(range(next_slot, self.npulay)) + list(range(0, next_slot))

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

    @staticmethod
    def _write_dataset(group, name: str, data) -> None:
        if name in group:
            del group[name]
        group.create_dataset(name, data=np.asarray(data, dtype=np.complex128))

    @staticmethod
    def _write_attrs(
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
    def _reset_component(comp_group) -> None:
        for name in ("last", "input_history", "residual_history"):
            if name in comp_group:
                del comp_group[name]
        comp_group.require_group("input_history")
        comp_group.require_group("residual_history")
