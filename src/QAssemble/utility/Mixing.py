import numpy as np


class Mixing:
    """Mixing schemes for self-consistent field iterations.

    Supports linear mixing and Pulay mixing (DIIS).
    """

    def __init__(self, method: str = "pulay", npulay: int = 5):
        """
        Parameters
        ----------
        method : str
            "linear" for simple linear mixing,
            "pulay" for Pulay/DIIS mixing.
        npulay : int
            Maximum number of history vectors kept for Pulay mixing.
        """
        self.method = method.lower()
        self.npulay = npulay
        self._input_history: list[np.ndarray] = []
        self._residual_history: list[np.ndarray] = []

    def reset(self):
        """Clear the mixing history."""
        self._input_history.clear()
        self._residual_history.clear()

    def __call__(
        self, iter: int, mix: float, Fnew: np.ndarray, Fold: np.ndarray
    ) -> np.ndarray:
        """Perform mixing.

        Parameters
        ----------
        iter : int
            Current SCF iteration number (1-indexed).
        mix : float
            Linear mixing parameter (0 < mix <= 1).
        Fnew : np.ndarray
            Output quantity from the current iteration.
        Fold : np.ndarray
            Input quantity fed into the current iteration.

        Returns
        -------
        np.ndarray
            Mixed quantity to be used as input for the next iteration.
        """
        if iter == 1:
            self.reset()
            return Fnew.copy()

        if self.method == "linear":
            return self._linear(mix, Fnew, Fold)
        elif self.method == "pulay":
            return self._pulay(mix, Fnew, Fold)
        else:
            raise ValueError(f"Unknown mixing method: {self.method}")

    @staticmethod
    def _linear(mix: float, Fnew: np.ndarray, Fold: np.ndarray) -> np.ndarray:
        return mix * Fnew + (1.0 - mix) * Fold

    def _pulay(
        self, mix: float, Fnew: np.ndarray, Fold: np.ndarray
    ) -> np.ndarray:
        """Pulay mixing (DIIS).

        The residual is defined as R = Fnew - Fold.
        We store input vectors (Fold) and residuals (R), then solve the
        DIIS linear system to find optimal coefficients that minimize
        the residual in a least-squares sense.

        The mixed input for the next iteration is:
            F_opt = sum_i c_i * (Fold_i + mix * R_i)
        """
        shape = Fnew.shape
        residual = Fnew - Fold

        self._input_history.append(Fold.ravel().copy())
        self._residual_history.append(residual.ravel().copy())

        # Trim history to npulay
        if len(self._input_history) > self.npulay:
            self._input_history.pop(0)
            self._residual_history.pop(0)

        nhist = len(self._input_history)

        if nhist == 1:
            # Not enough history yet; fall back to linear mixing
            return self._linear(mix, Fnew, Fold)

        # Build the overlap matrix: C[i,j] = Re(<R_i | R_j>)
        C = np.zeros((nhist, nhist), dtype=np.float64)
        for i in range(nhist):
            for j in range(i, nhist):
                Cij = np.vdot(self._residual_history[i], self._residual_history[j]).real
                C[i, j] = Cij
                C[j, i] = Cij

        # Obtain optimal coefficients via inverse of C, then normalize
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            # Singular matrix; fall back to linear mixing
            return self._linear(mix, Fnew, Fold)

        c = C_inv.sum(axis=1)
        c_sum = c.sum()
        if abs(c_sum) < 1.0e-14:
            return self._linear(mix, Fnew, Fold)
        c /= c_sum

        # Construct the optimal mixed vector
        Fopt = np.zeros_like(self._input_history[0])
        for i in range(nhist):
            Fopt += c[i] * (self._input_history[i] + mix * self._residual_history[i])

        return Fopt.reshape(shape)
