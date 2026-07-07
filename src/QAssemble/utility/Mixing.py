import numpy as np


class Mixing:
    """Stateless linear/Pulay mixer.

    HDF5 state management is handled by ``utility.HDF5.IO``.  This class only
    computes a mixed array from the current value, previous value, and optional
    Pulay history.
    """

    def __call__(
        self,
        method: str = "pulay",
        mix: float = None,
        fnew=None,
        fold=None,
        inputs=None,
        residuals=None,
    ) -> np.ndarray:
        method = str(method).lower()
        if method == "linear":
            return self.linear(mix=mix, fnew=fnew, fold=fold)
        if method == "pulay":
            return self.pulay(
                mix=mix,
                fnew=fnew,
                fold=fold,
                inputs=inputs,
                residuals=residuals,
            )
        raise ValueError(f"Unknown mixing method: {method}")

    @staticmethod
    def linear(mix: float, fnew, fold) -> np.ndarray:
        mix = Mixing._validate_mix(mix)
        fnew = Mixing._as_array("fnew", fnew)
        fold = Mixing._as_array("fold", fold)
        Mixing._validate_same_shape("fold", fold, fnew)
        return np.asfortranarray(mix * fnew + (1.0 - mix) * fold)

    @staticmethod
    def pulay(
        mix: float,
        fnew,
        fold,
        inputs=None,
        residuals=None,
    ) -> np.ndarray:
        mix = Mixing._validate_mix(mix)
        fnew = Mixing._as_array("fnew", fnew)
        fold = Mixing._as_array("fold", fold)
        Mixing._validate_same_shape("fold", fold, fnew)

        inputs = [] if inputs is None else list(inputs)
        residuals = [] if residuals is None else list(residuals)
        if len(inputs) != len(residuals):
            raise ValueError("inputs and residuals must have the same length")
        if len(inputs) < 2:
            return Mixing.linear(mix=mix, fnew=fnew, fold=fold)

        inputs = [Mixing._as_array(f"inputs[{idx}]", arr) for idx, arr in enumerate(inputs)]
        residuals = [
            Mixing._as_array(f"residuals[{idx}]", arr)
            for idx, arr in enumerate(residuals)
        ]
        for idx, arr in enumerate(inputs):
            Mixing._validate_same_shape(f"inputs[{idx}]", arr, fnew)
        for idx, arr in enumerate(residuals):
            Mixing._validate_same_shape(f"residuals[{idx}]", arr, fnew)

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
            return Mixing.linear(mix=mix, fnew=fnew, fold=fold)

        coeff_sum = coeff.sum()
        if abs(coeff_sum) < 1.0e-14:
            return Mixing.linear(mix=mix, fnew=fnew, fold=fold)
        coeff /= coeff_sum

        mixed_flat = np.zeros(inputs[0].size, dtype=np.complex128)
        for ci, input_i, residual_i in zip(coeff, inputs, residuals):
            mixed_flat += ci * (input_i.ravel() + mix * residual_i.ravel())

        return np.asfortranarray(mixed_flat.reshape(fnew.shape))

    @staticmethod
    def _validate_mix(mix: float) -> float:
        if mix is None:
            raise ValueError("Mixing requires mix")
        mix = float(mix)
        if not (0.0 < mix <= 1.0):
            raise ValueError("mix must satisfy 0 < mix <= 1")
        return mix

    @staticmethod
    def _as_array(name: str, value) -> np.ndarray:
        if value is None:
            raise ValueError(f"{name} is required")
        arr = np.asarray(value, dtype=np.complex128)
        if arr.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
            raise ValueError(f"{name} contains non-finite values")
        return np.asfortranarray(arr)

    @staticmethod
    def _validate_same_shape(name: str, arr: np.ndarray, ref: np.ndarray) -> None:
        if arr.shape != ref.shape:
            raise ValueError(f"{name} shape mismatch: expected {ref.shape}, got {arr.shape}")
