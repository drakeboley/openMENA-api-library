"""
OpenMENA API — Python interface for the Open Memristor-in-Memory Accelerator.

Based on the OpenMENA platform (arXiv:2511.03747), this module provides a high-level
simulation of a memristor crossbar array using the VIPI (Voltage-Incremental
Proportional-Integral) programming method. Designed for neuromorphic edge-AI
research including inference, on-device learning, and chip-in-the-loop fine-tuning.

Key references:
  - OpenMENA: Safa et al., "OpenMENA: An Open-Source Memristor Interfacing and
    Compute Board for Neuromorphic Edge-AI Applications", arXiv:2511.03747, 2025.
  - VIPI: Voltage-Incremental Proportional-Integral control for analog conductance
    programming with closed-loop write-verify feedback.

Example:
    >>> board = OpenMenaBoard(rows=64, cols=10)
    >>> board.weight_transfer(trained_weights)
    >>> logits = board.matrix_vector_multiply(input_image)
    >>> prediction = np.argmax(board.softmax(logits))
"""

import numpy as np
import struct
import math
from typing import List, Tuple, Union, Callable, Dict, Optional


class OpenMenaBoard:
    """
    Simulates an OpenMENA (Open Memristor-in-Memory Accelerator) board.

    The crossbar is an (rows × cols) array of memristive devices, each storing
    an analog conductance G ∈ [G_min, G_max]. Input voltages V are applied to
    rows, and output currents I are read from columns via Kirchhoff's current law:

        I_j = Σᵢ G(i,j) · V_i        (Matrix-Vector Multiply)

    This class provides:
      - Core device operations (forming, set, reset, read)
      - VIPI closed-loop programming algorithm
      - In-memory matrix-vector multiplication
      - Neural network inference and on-device learning
      - Hardware metrics (power, energy, SNR, endurance, retention)
      - Binary file I/O (.bin state save/load)
      - Noise injection and non-ideality simulation

    Attributes:
        rows (int): Number of rows in the crossbar.
        cols (int): Number of columns in the crossbar.
        g_min (float): Minimum conductance in Siemens (HRS).
        g_max (float): Maximum conductance in Siemens (LRS).
        crossbar (np.ndarray): The conductance matrix, shape (rows, cols).
        formed (np.ndarray): Boolean mask of formed (initialized) cells.
        cycles (np.ndarray): Endurance cycle count per cell.
        stuck_at_faults (np.ndarray): Fault map (0=normal, 1=stuck-low, 2=stuck-high).
        programming_log (list): History of programming operations.
    """

    # ================================================================
    #  Construction & Configuration
    # ================================================================

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        g_min: float = 1e-6,
        g_max: float = 1e-3,
        read_voltage: float = 0.1,
        noise_sigma: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Initialize the OpenMENA board simulator.

        Args:
            rows: Number of rows (input lines) in the crossbar.
            cols: Number of columns (output lines) in the crossbar.
            g_min: Minimum conductance (High Resistance State, HRS) in Siemens.
                   Typical value: 1 µS (1e-6 S).
            g_max: Maximum conductance (Low Resistance State, LRS) in Siemens.
                   Typical value: 1 mS (1e-3 S).
            read_voltage: Default read pulse amplitude in Volts (below switching threshold).
            noise_sigma: Relative noise standard deviation for read operations.
            seed: Random seed for reproducibility.
        """
        self.rows = rows
        self.cols = cols
        self.g_min = g_min
        self.g_max = g_max
        self.g_range = g_max - g_min
        self.read_voltage = read_voltage
        self.noise_sigma = noise_sigma

        self.rng = np.random.default_rng(seed)

        # Crossbar state
        self.crossbar = np.full((rows, cols), g_min, dtype=np.float64)
        self.formed = np.zeros((rows, cols), dtype=bool)
        self.cycles = np.zeros((rows, cols), dtype=np.int64)
        self.stuck_at_faults = np.zeros((rows, cols), dtype=np.int8)

        # Telemetry
        self.programming_log: List[Dict] = []

    # ================================================================
    #  Core Memristor Device Operations
    # ================================================================

    def forming(self, row: int, col: int, voltage: float = 3.0) -> float:
        """
        Electroforming — create the initial conductive filament in a pristine device.

        Electroforming is a one-time, high-voltage process that creates a nanoscale
        conductive filament (typically of oxygen vacancies in metal-oxide memristors)
        connecting the top and bottom electrodes. After forming, the device can be
        reversibly switched between HRS and LRS.

        Physics model:
            G_initial ≈ G_min × 10 + stochastic variation

        Args:
            row: Row index.
            col: Column index.
            voltage: Forming voltage in Volts (typically 2–5V, must be ≥ 2V).

        Returns:
            The initial post-forming conductance.

        Raises:
            ValueError: If voltage is below the forming threshold.
        """
        if voltage < 2.0:
            raise ValueError(f"Forming voltage {voltage}V too low (minimum 2V)")
        if self._is_stuck(row, col):
            return self.crossbar[row, col]

        g_initial = self.g_min * 10.0 + self.g_min * abs(self.rng.normal(0, 1)) * 100.0
        g_initial = np.clip(g_initial, self.g_min, self.g_max)
        self.crossbar[row, col] = g_initial
        self.formed[row, col] = True
        self.cycles[row, col] += 1
        self._log("FORM", row, col, g_initial)
        return g_initial

    def form_all(self, voltage: float = 3.0) -> None:
        """Electroform every cell in the array."""
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.formed[r, c]:
                    self.forming(r, c, voltage)

    def set_conductance(self, row: int, col: int, target_g: float) -> float:
        """
        SET operation — program a device toward LRS (increase conductance).

        Applies a positive voltage pulse to transition the memristor from HRS
        to a target conductance. The actual achieved conductance includes
        stochastic switching noise.

        Model:
            G_actual = target_g × (1 + ε),   ε ~ N(0, σ²)

        Args:
            row: Row index.
            col: Column index.
            target_g: Target conductance in Siemens.

        Returns:
            Actual conductance achieved.
        """
        if self._is_stuck(row, col):
            return self.crossbar[row, col]
        if not self.formed[row, col]:
            self.forming(row, col)

        noise_factor = 1.0 + self.rng.normal(0, self.noise_sigma * 2)
        g_actual = np.clip(target_g * noise_factor, self.g_min, self.g_max)
        self.crossbar[row, col] = g_actual
        self.cycles[row, col] += 1
        self._log("SET", row, col, g_actual)
        return g_actual

    def reset_conductance(self, row: int, col: int) -> float:
        """
        RESET operation — return a device to HRS (minimum conductance).

        Applies a negative (or opposite polarity) voltage pulse to rupture the
        conductive filament and return the device to its high-resistance state.

        Returns:
            Conductance after reset (approximately G_min).
        """
        if self._is_stuck(row, col):
            return self.crossbar[row, col]

        g_reset = self.g_min * (1.0 + abs(self.rng.normal(0, self.noise_sigma)))
        g_reset = np.clip(g_reset, self.g_min, self.g_max)
        self.crossbar[row, col] = g_reset
        self.cycles[row, col] += 1
        self._log("RESET", row, col, g_reset)
        return g_reset

    def reset_all(self) -> None:
        """Reset every cell to HRS."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.reset_conductance(r, c)

    def read_conductance(self, row: int, col: int) -> float:
        """
        Read the conductance of a single device.

        Applies a small read voltage (below switching threshold) and measures
        the resulting current: G = I / V_read. Includes read noise.

        Model:
            G_measured = G_actual × (1 + ε),   ε ~ N(0, σ²)

        Returns:
            Measured conductance in Siemens.
        """
        g = self.crossbar[row, col]
        noise = g * self.rng.normal(0, self.noise_sigma)
        return float(np.clip(g + noise, self.g_min, self.g_max))

    def read_all(self) -> np.ndarray:
        """
        Read the entire crossbar conductance matrix (with noise).

        Returns:
            np.ndarray of shape (rows, cols) with measured conductances.
        """
        noise = self.crossbar * self.rng.normal(0, self.noise_sigma, size=self.crossbar.shape)
        return np.clip(self.crossbar + noise, self.g_min, self.g_max)

    def apply_voltage_pulse(
        self, row: int, col: int, voltage: float, width_us: float = 1.0
    ) -> float:
        """
        Apply a raw voltage pulse to modify conductance.

        The conductance change depends on pulse amplitude and duration following
        a simplified filament growth/dissolution model:

            ΔG = α · V · t_pulse · (G_max - G) / G_range     (for V > 0, SET)
            ΔG = α · V · t_pulse · (G - G_min) / G_range     (for V < 0, RESET)

        where α is a device-dependent rate constant.

        Args:
            row: Row index.
            col: Column index.
            voltage: Pulse amplitude in Volts (+ve for SET, -ve for RESET).
            width_us: Pulse width in microseconds.

        Returns:
            New conductance after pulse.
        """
        if self._is_stuck(row, col):
            return self.crossbar[row, col]

        g = self.crossbar[row, col]
        alpha = 1e-4  # rate constant
        if voltage > 0:
            headroom = (self.g_max - g) / self.g_range
        else:
            headroom = (g - self.g_min) / self.g_range

        delta_g = alpha * voltage * width_us * headroom
        g_new = np.clip(g + delta_g, self.g_min, self.g_max)
        self.crossbar[row, col] = g_new
        self.cycles[row, col] += 1
        return float(g_new)

    # ================================================================
    #  VIPI Programming — Voltage-Incremental Proportional-Integral
    # ================================================================

    def vipi_program(
        self,
        row: int,
        col: int,
        target_g: float,
        kp: float = 0.5,
        ki: float = 0.1,
        v_increment: float = 0.05,
        max_iterations: int = 50,
        tolerance: float = 0.05,
    ) -> Dict:
        """
        VIPI closed-loop programming algorithm for a single memristor cell.

        The Voltage-Incremental Proportional-Integral (VIPI) method from OpenMENA
        iteratively programs a device to a target conductance using PI control with
        adaptive voltage stepping. It outperforms simple write-verify by accounting
        for device-to-device threshold voltage variability.

        Algorithm:
            1. Measure current conductance G_k
            2. Compute error: e_k = G_target - G_k
            3. If |e_k / G_target| < tolerance → converged
            4. Accumulate integral: I_k = I_{k-1} + e_k
            5. PI correction: u_k = Kp · e_k + Ki · I_k
            6. Update voltage: V_k += V_inc · sign(e_k)
            7. Apply pulse: ΔG ∝ u_k · V_k
            8. Repeat from step 1

        Args:
            row: Row index.
            col: Column index.
            target_g: Target conductance in Siemens.
            kp: Proportional gain of the PI controller.
            ki: Integral gain of the PI controller.
            v_increment: Voltage step size in Volts per iteration.
            max_iterations: Maximum programming iterations.
            tolerance: Relative error tolerance for convergence.

        Returns:
            Dict with keys: 'converged' (bool), 'iterations' (int),
            'final_g' (float), 'final_error' (float), 'history' (list).
        """
        if not self.formed[row, col]:
            self.forming(row, col)

        target_g = np.clip(target_g, self.g_min, self.g_max)
        integral_error = 0.0
        voltage = 0.5
        history = []

        for i in range(max_iterations):
            current_g = self.read_conductance(row, col)
            error = target_g - current_g
            rel_error = abs(error / target_g) if target_g > 0 else abs(error)

            history.append({
                "iter": i, "g": current_g, "target": target_g,
                "error": error, "voltage": voltage
            })

            if rel_error < tolerance:
                self._log("VIPI_OK", row, col, current_g)
                return {
                    "converged": True, "iterations": i,
                    "final_g": current_g, "final_error": error, "history": history
                }

            # PI control
            integral_error += error
            correction = kp * error + ki * integral_error

            # Voltage-incremental adaptation
            voltage += v_increment * (1.0 if error > 0 else -1.0)
            voltage = np.clip(voltage, 0.1, 3.0)

            # Apply programming pulse
            delta_g = correction * voltage * 1e-4
            self.crossbar[row, col] = np.clip(
                self.crossbar[row, col] + delta_g, self.g_min, self.g_max
            )
            self.cycles[row, col] += 1

        final_g = self.read_conductance(row, col)
        self._log("VIPI_FAIL", row, col, final_g)
        return {
            "converged": False, "iterations": max_iterations,
            "final_g": final_g, "final_error": target_g - final_g, "history": history
        }

    def vipi_program_matrix(
        self,
        target_matrix: np.ndarray,
        kp: float = 0.5,
        ki: float = 0.1,
        v_increment: float = 0.05,
        max_iterations: int = 50,
        tolerance: float = 0.05,
    ) -> Dict:
        """
        Program the entire crossbar array using VIPI.

        Args:
            target_matrix: Target conductance matrix, shape (rows, cols).
            kp, ki, v_increment, max_iterations, tolerance: VIPI parameters.

        Returns:
            Dict with 'yield_pct' (float), 'total_cycles' (int),
            'converged_count' (int), 'failed_cells' (list of (r,c) tuples).
        """
        if target_matrix.shape != (self.rows, self.cols):
            raise ValueError(
                f"Target matrix shape {target_matrix.shape} != crossbar ({self.rows},{self.cols})"
            )

        converged = 0
        failed = []
        for r in range(self.rows):
            for c in range(self.cols):
                result = self.vipi_program(
                    r, c, target_matrix[r, c], kp, ki, v_increment, max_iterations, tolerance
                )
                if result["converged"]:
                    converged += 1
                else:
                    failed.append((r, c))

        total = self.rows * self.cols
        return {
            "yield_pct": converged / total * 100,
            "total_cycles": int(self.cycles.sum()),
            "converged_count": converged,
            "failed_cells": failed,
        }

    def write_verify(
        self, row: int, col: int, target_g: float,
        tolerance: float = 0.05, max_attempts: int = 20
    ) -> bool:
        """
        Simple write-verify loop (non-VIPI baseline).

        Repeatedly writes the target conductance and reads back until the
        measured value is within tolerance. Less accurate than VIPI for
        devices with variable threshold voltages.

        Args:
            row, col: Cell coordinates.
            target_g: Target conductance.
            tolerance: Relative tolerance.
            max_attempts: Maximum write-read cycles.

        Returns:
            True if the cell converged within tolerance.
        """
        for _ in range(max_attempts):
            self.set_conductance(row, col, target_g)
            measured = self.read_conductance(row, col)
            if abs(measured - target_g) / target_g < tolerance:
                return True
        return False

    # ================================================================
    #  In-Memory Computation
    # ================================================================

    def matrix_vector_multiply(self, input_vector: np.ndarray) -> np.ndarray:
        """
        In-memory matrix-vector multiplication via Ohm's law and KCL.

        Each input voltage V_i is applied to row i. The current at column j is:

            I_j = Σᵢ G(i,j) · V_i

        This is the fundamental compute primitive of memristive crossbar arrays,
        performing an entire matrix-vector product in O(1) time steps.

        Args:
            input_vector: Input voltages, shape (rows,).

        Returns:
            Output currents, shape (cols,).
        """
        if len(input_vector) != self.rows:
            raise ValueError(f"Input length {len(input_vector)} != rows {self.rows}")
        G = self.read_all()
        return G.T @ input_vector

    def differential_pair_mvm(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiply using differential pair encoding for signed weights.

        Since memristor conductances are inherently non-negative, signed weights
        are represented using two devices per weight:

            W(i,j) = G⁺(i,j) - G⁻(i,j)

        The crossbar is split: first half of columns = G⁺, second half = G⁻.

        Args:
            input_vector: Input voltages, shape (rows,).

        Returns:
            Output currents (signed), shape (cols//2,).
        """
        half = self.cols // 2
        G = self.read_all()
        g_pos = G[:, :half]
        g_neg = G[:, half:]
        return (g_pos - g_neg).T @ input_vector

    def current_summation(self, input_voltages: np.ndarray) -> np.ndarray:
        """
        Kirchhoff's current law summation at column bit-lines.

        Alias for matrix_vector_multiply emphasizing the physical operation.
        """
        return self.matrix_vector_multiply(input_voltages)

    def batch_mvm(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Batched matrix-vector multiplication.

        Args:
            input_batch: Input voltage matrix, shape (batch_size, rows).

        Returns:
            Output current matrix, shape (batch_size, cols).
        """
        G = self.read_all()
        return input_batch @ G

    # ================================================================
    #  Activation Functions
    # ================================================================

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit: f(x) = max(0, x)."""
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU: f(x) = x if x > 0, else αx."""
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Logistic sigmoid: f(x) = 1 / (1 + exp(-x))."""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def tanh_activation(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent: f(x) = tanh(x)."""
        return np.tanh(x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax function: σ(x)_j = exp(x_j) / Σₖ exp(x_k).

        Numerically stable implementation with max subtraction.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish / SiLU activation: f(x) = x · σ(x)."""
        return x * OpenMenaBoard.sigmoid(x)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit (approximate)."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    # ================================================================
    #  Neural Network Inference
    # ================================================================

    def inference(self, input_data: np.ndarray, activation_fn: Callable) -> np.ndarray:
        """
        Single-layer neural network forward pass.

            output = activation(G^T · input)

        Args:
            input_data: Input vector, shape (rows,).
            activation_fn: Activation function (e.g., self.relu).

        Returns:
            Activated output, shape (cols,).
        """
        raw = self.matrix_vector_multiply(input_data)
        return activation_fn(raw)

    def multi_layer_inference(
        self,
        input_data: np.ndarray,
        weight_matrices: List[np.ndarray],
        activation_fns: List[Callable],
    ) -> np.ndarray:
        """
        Multi-layer neural network forward pass (simulated off-crossbar).

        For each layer l:
            h_l = activation_l(W_l^T · h_{l-1})

        Args:
            input_data: Network input.
            weight_matrices: List of weight matrices per layer.
            activation_fns: List of activation functions per layer.

        Returns:
            Network output.
        """
        x = input_data.copy()
        for W, act in zip(weight_matrices, activation_fns):
            x = act(W.T @ x)
        return x

    def batch_inference(
        self, batch_data: np.ndarray, activation_fn: Callable
    ) -> np.ndarray:
        """
        Batch inference through the crossbar.

        Args:
            batch_data: Input batch, shape (batch_size, rows).
            activation_fn: Activation function.

        Returns:
            Activated outputs, shape (batch_size, cols).
        """
        return activation_fn(self.batch_mvm(batch_data))

    # ================================================================
    #  Loss Functions & Metrics
    # ================================================================

    @staticmethod
    def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Categorical cross-entropy loss:
            L = -Σ t_j · log(p_j)
        """
        predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        return -float(np.sum(targets * np.log(predictions)))

    @staticmethod
    def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Binary cross-entropy loss:
            L = -Σ [t·log(p) + (1-t)·log(1-p)]
        """
        p = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        return -float(np.sum(targets * np.log(p) + (1 - targets) * np.log(1 - p)))

    @staticmethod
    def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Squared Error: L = (1/n) Σ (p - t)²."""
        return float(np.mean((predictions - targets) ** 2))

    @staticmethod
    def mae_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Absolute Error: L = (1/n) Σ |p - t|."""
        return float(np.mean(np.abs(predictions - targets)))

    @staticmethod
    def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Classification accuracy (argmax-based)."""
        pred_cls = np.argmax(predictions, axis=-1)
        true_cls = np.argmax(labels, axis=-1) if labels.ndim > 1 else labels
        return float(np.mean(pred_cls == true_cls))

    @staticmethod
    def confusion_matrix(
        predictions: np.ndarray, labels: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """
        Compute the confusion matrix.

        Returns:
            np.ndarray of shape (num_classes, num_classes) where entry (i,j) is the
            count of samples with true class i predicted as class j.
        """
        pred_cls = np.argmax(predictions, axis=-1)
        true_cls = np.argmax(labels, axis=-1) if labels.ndim > 1 else labels
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(true_cls, pred_cls):
            cm[t, p] += 1
        return cm

    @staticmethod
    def precision_recall_f1(
        predictions: np.ndarray, labels: np.ndarray, num_classes: int
    ) -> Dict[str, np.ndarray]:
        """
        Per-class precision, recall, and F1 score.

        Returns:
            Dict with 'precision', 'recall', 'f1' arrays of shape (num_classes,).
        """
        cm = OpenMenaBoard.confusion_matrix(predictions, labels, num_classes)
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        for i in range(num_classes):
            col_sum = cm[:, i].sum()
            row_sum = cm[i, :].sum()
            precision[i] = cm[i, i] / col_sum if col_sum > 0 else 0
            recall[i] = cm[i, i] / row_sum if row_sum > 0 else 0
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0,
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    # ================================================================
    #  On-Device Learning & Weight Transfer
    # ================================================================

    def weight_transfer(self, trained_weights: np.ndarray) -> Dict:
        """
        Transfer pre-trained neural network weights to the crossbar via VIPI.

        Normalizes weights to the conductance range and programs each cell.

        Args:
            trained_weights: Weight matrix, shape (rows, cols). Values in [-1, 1]
                             are mapped to [G_min, G_max].

        Returns:
            VIPI programming report dict.
        """
        target_g = self.weight_to_conductance(trained_weights)
        return self.vipi_program_matrix(target_g)

    def on_device_learning(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 10,
        activation_fn: Optional[Callable] = None,
    ) -> List[float]:
        """
        On-device training using the delta (Widrow-Hoff) rule.

        For each sample (x, y):
            1. Forward pass:  ŷ = activation(G^T · x)
            2. Compute error: e = y - ŷ
            3. Weight update:  ΔW = η · x ⊗ e     (outer product)
            4. Reprogram crossbar conductances

        This is the "chip-in-the-loop" training loop where the physical crossbar
        state is read and updated each iteration.

        Args:
            inputs: Training inputs, shape (n_samples, rows).
            targets: Training targets, shape (n_samples, cols).
            learning_rate: Learning rate η.
            epochs: Number of training epochs.
            activation_fn: Activation function (default: sigmoid).

        Returns:
            List of per-epoch average losses.
        """
        if activation_fn is None:
            activation_fn = self.sigmoid

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in zip(inputs, targets):
                # Forward pass through physical crossbar
                y_hat = self.inference(x, activation_fn)
                error = y - y_hat
                epoch_loss += self.mse_loss(y_hat, y)

                # Delta rule update
                dW = learning_rate * np.outer(x, error)
                current_w = self.conductance_to_weight(self.read_all())
                new_w = np.clip(current_w + dW, -1, 1)
                new_g = self.weight_to_conductance(new_w)

                # Reprogram (fast mode — direct set for training speed)
                for r in range(self.rows):
                    for c in range(self.cols):
                        self.set_conductance(r, c, new_g[r, c])

            losses.append(epoch_loss / len(inputs))
        return losses

    def chip_in_the_loop_finetune(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        lr: float = 0.001,
        epochs: int = 5,
    ) -> List[float]:
        """
        Chip-in-the-loop fine-tuning to compensate for hardware non-idealities.

        After initial weight transfer, this method reads the actual crossbar state
        (including noise, drift, and stuck faults) and fine-tunes the weights
        to recover accuracy lost to device imperfections.

        Uses VIPI programming for weight updates (slower but more accurate than
        direct SET used in on_device_learning).

        Returns:
            List of per-epoch average losses.
        """
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in zip(inputs, targets):
                y_hat = self.inference(x, self.sigmoid)
                error = y - y_hat
                epoch_loss += self.mse_loss(y_hat, y)

                dW = lr * np.outer(x, error)
                current_w = self.conductance_to_weight(self.read_all())
                new_w = np.clip(current_w + dW, -1, 1)
                new_g = self.weight_to_conductance(new_w)

                # Use VIPI for accurate programming
                self.vipi_program_matrix(new_g, tolerance=0.02)

            losses.append(epoch_loss / len(inputs))
        return losses

    # ================================================================
    #  Weight ↔ Conductance Mapping
    # ================================================================

    def conductance_to_weight(self, g: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Map conductance G ∈ [G_min, G_max] to weight w ∈ [-1, 1].

            w = 2 · (G - G_min) / (G_max - G_min) - 1
        """
        return 2.0 * (g - self.g_min) / self.g_range - 1.0

    def weight_to_conductance(self, w: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Map weight w ∈ [-1, 1] to conductance G ∈ [G_min, G_max].

            G = G_min + (w + 1) / 2 · (G_max - G_min)
        """
        return self.g_min + (np.asarray(w) + 1.0) / 2.0 * self.g_range

    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weight matrix to [-1, 1] range for conductance mapping."""
        w_max = np.max(np.abs(weights))
        return weights / w_max if w_max > 0 else weights

    def quantize_weights(self, weights: np.ndarray, levels: int) -> np.ndarray:
        """
        Quantize weights to discrete conductance levels.

        Memristors have a finite number of distinguishable states. This function
        maps continuous weights to the nearest discrete level.

        Args:
            weights: Weight matrix.
            levels: Number of discrete conductance levels (e.g., 16, 32, 64).

        Returns:
            Quantized weight matrix.
        """
        return np.round(weights * (levels - 1)) / (levels - 1)

    @staticmethod
    def map_signed_weights(weight_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose signed weights into differential pair (G⁺, G⁻).

            W = G⁺ - G⁻
            G⁺ = max(0, W)
            G⁻ = max(0, -W)

        Returns:
            Tuple of (g_plus, g_minus) arrays.
        """
        return np.maximum(0, weight_matrix), np.maximum(0, -weight_matrix)

    # ================================================================
    #  Noise & Non-Ideality Simulation
    # ================================================================

    def add_device_noise(self, sigma: float = 0.01) -> None:
        """
        Inject Gaussian device-to-device variability into conductances.

        Models manufacturing process variations that cause each device to
        deviate slightly from its programmed conductance.

        Args:
            sigma: Noise standard deviation in Siemens.
        """
        noise = self.rng.normal(0, sigma, size=self.crossbar.shape)
        self.crossbar = np.clip(self.crossbar + noise, self.g_min, self.g_max)

    def add_read_noise(self, sigma_relative: float = 0.05) -> np.ndarray:
        """
        Return a noisy snapshot of the crossbar (non-destructive).

        Args:
            sigma_relative: Relative noise standard deviation.

        Returns:
            Noisy conductance matrix (does not modify internal state).
        """
        noise = self.crossbar * self.rng.normal(0, sigma_relative, size=self.crossbar.shape)
        return np.clip(self.crossbar + noise, self.g_min, self.g_max)

    def apply_stuck_at_faults(self, fault_rate: float = 0.01) -> int:
        """
        Simulate manufacturing defects (stuck-at faults).

        A fraction of cells are permanently stuck at either G_min (SA0)
        or G_max (SA1), modelling yield loss in fabrication.

        Args:
            fault_rate: Fraction of cells to make faulty (0 to 1).

        Returns:
            Number of faults injected.
        """
        n_faults = int(self.rows * self.cols * fault_rate)
        indices = self.rng.choice(self.rows * self.cols, n_faults, replace=False)
        for idx in indices:
            r, c = divmod(int(idx), self.cols)
            fault_type = self.rng.choice([1, 2])
            self.stuck_at_faults[r, c] = fault_type
            self.crossbar[r, c] = self.g_min if fault_type == 1 else self.g_max
        return n_faults

    def apply_wire_resistance(self, r_wire: float = 1.0) -> np.ndarray:
        """
        Simulate parasitic interconnect (wire) resistance in the crossbar.

        In real crossbar arrays, each row and column line has a finite resistance
        that causes IR drops, reducing the effective voltage seen by distant cells.

        Model: V_eff(i,j) = V_in(i) - i · R_wire · I_accumulated

        Args:
            r_wire: Wire resistance per segment in Ohms.

        Returns:
            Effective conductance matrix after wire resistance compensation.
        """
        G_eff = self.crossbar.copy()
        for r in range(self.rows):
            cumulative_drop = 0.0
            for c in range(self.cols):
                # Current through this cell
                i_cell = G_eff[r, c] * self.read_voltage
                cumulative_drop += i_cell * r_wire
                # Reduce effective conductance due to voltage drop
                v_effective = max(0, self.read_voltage - cumulative_drop)
                if self.read_voltage > 0:
                    G_eff[r, c] *= v_effective / self.read_voltage
        return G_eff

    def apply_sneak_path_current(self) -> np.ndarray:
        """
        Estimate sneak path current contamination.

        In passive crossbar arrays (without selectors), unselected cells create
        parasitic current paths that add noise to the readout. This method
        estimates the worst-case sneak current for each column.

        Returns:
            Array of sneak current magnitudes per column, shape (cols,).
        """
        sneak = np.zeros(self.cols)
        for c in range(self.cols):
            # Worst case: all other rows at V_read, all other cols at 0V
            for r in range(self.rows):
                # Sneak current through parallel paths
                parallel_g = np.sum(self.crossbar[r, :]) - self.crossbar[r, c]
                if parallel_g > 0:
                    sneak[c] += self.crossbar[r, c] * self.read_voltage * 0.01  # Approximation
        return sneak

    # ================================================================
    #  Hardware Metrics & Diagnostics
    # ================================================================

    def snr(self) -> float:
        """
        Signal-to-noise ratio of the crossbar in dB.

            SNR = 10 · log₁₀(P_signal / P_noise)

        where P_signal = mean(G²) and P_noise is estimated from read noise.
        """
        signal_power = np.mean(self.crossbar ** 2)
        noise_power = np.mean((self.crossbar * self.noise_sigma) ** 2)
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

    def conductance_histogram(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distribution of conductance values across the array.

        Returns:
            Tuple of (counts, bin_edges).
        """
        return np.histogram(self.crossbar.ravel(), bins=bins)

    def power_consumption(self, input_vector: np.ndarray) -> float:
        """
        Estimate static power consumption for an MVM operation.

            P = Σᵢ Σⱼ V_i² · G(i,j)

        Args:
            input_vector: Input voltages, shape (rows,).

        Returns:
            Power in Watts.
        """
        V2 = input_vector ** 2
        return float(np.sum(V2[:, np.newaxis] * self.crossbar))

    def energy_per_mac(self, v_read: float = 0.1, t_read_ns: float = 100.0) -> float:
        """
        Energy per multiply-accumulate operation.

            E_MAC = V_read² · G_avg · t_read

        Args:
            v_read: Read voltage in Volts.
            t_read_ns: Read pulse duration in nanoseconds.

        Returns:
            Energy per MAC in Joules.
        """
        avg_g = np.mean(self.crossbar)
        return v_read ** 2 * avg_g * (t_read_ns * 1e-9)

    def throughput_tops(self, clock_freq_mhz: float = 100.0) -> float:
        """
        Estimated throughput in Tera Operations Per Second (TOPS).

        Each clock cycle performs rows × cols MAC operations in parallel.

        Args:
            clock_freq_mhz: Operating clock frequency in MHz.

        Returns:
            TOPS.
        """
        ops_per_cycle = self.rows * self.cols * 2  # multiply + accumulate
        return ops_per_cycle * clock_freq_mhz * 1e6 / 1e12

    def energy_efficiency_tops_per_w(
        self, v_read: float = 0.1, t_read_ns: float = 100.0
    ) -> float:
        """
        Energy efficiency in TOPS/W.

        Returns:
            TOPS per Watt.
        """
        ops_per_cycle = self.rows * self.cols * 2
        energy_per_cycle = self.power_consumption(np.full(self.rows, v_read)) * t_read_ns * 1e-9
        if energy_per_cycle <= 0:
            return float("inf")
        return ops_per_cycle / energy_per_cycle / 1e12

    def crossbar_utilization(self) -> float:
        """Percentage of cells with conductance significantly above G_min."""
        threshold = self.g_min * 1.5
        return float(np.sum(self.crossbar > threshold) / (self.rows * self.cols) * 100)

    def programming_yield(self) -> float:
        """Percentage of non-faulty cells."""
        return float(np.sum(self.stuck_at_faults == 0) / (self.rows * self.cols) * 100)

    def endurance_test(self, row: int, col: int, cycles: int = 1000) -> Dict:
        """
        Simulate endurance cycling and potential degradation.

        Memristors have finite write endurance (typically 10⁶–10¹² cycles).
        After many cycles, devices may develop stuck-at faults.

        Args:
            row, col: Cell coordinates.
            cycles: Number of SET/RESET cycles to simulate.

        Returns:
            Dict with 'total_cycles', 'degraded' (bool), 'fault' (bool).
        """
        self.cycles[row, col] += cycles
        degraded = False
        faulted = False

        if self.cycles[row, col] > 1e6:
            # Add conductance drift proportional to cycle count
            drift = self.rng.normal(0, self.noise_sigma * (self.cycles[row, col] / 1e6))
            self.crossbar[row, col] = np.clip(
                self.crossbar[row, col] + drift, self.g_min, self.g_max
            )
            degraded = True

        if self.cycles[row, col] > 1e8:
            if self.rng.random() > 0.9:
                self.stuck_at_faults[row, col] = self.rng.choice([1, 2])
                faulted = True

        return {
            "total_cycles": int(self.cycles[row, col]),
            "degraded": degraded,
            "fault": faulted,
        }

    def retention_test(self, row: int, col: int, time_seconds: float) -> float:
        """
        Simulate conductance drift over time (retention loss).

        Memristor conductances drift over time due to diffusion of ions/vacancies.
        Model: G(t) = G_min + (G_0 - G_min) · exp(-t / τ)

        Args:
            row, col: Cell coordinates.
            time_seconds: Elapsed time in seconds.

        Returns:
            Conductance after drift.
        """
        if self._is_stuck(row, col):
            return self.crossbar[row, col]

        tau = 1e5  # Retention time constant (100,000 seconds ≈ 28 hours)
        g0 = self.crossbar[row, col]
        g_new = self.g_min + (g0 - self.g_min) * np.exp(-time_seconds / tau)
        self.crossbar[row, col] = g_new
        return float(g_new)

    @staticmethod
    def von_neumann_bottleneck_speedup(
        traditional_ops: int, crossbar_ops: int
    ) -> float:
        """
        Estimate speedup of in-memory compute over von Neumann architecture.

        In von Neumann systems, data must be fetched from memory to the CPU
        for each operation. Crossbar arrays compute in-place, eliminating
        the memory bottleneck.

        Args:
            traditional_ops: Number of operations in traditional architecture.
            crossbar_ops: Number of operations (cycles) on the crossbar.

        Returns:
            Speedup factor.
        """
        return traditional_ops / crossbar_ops if crossbar_ops > 0 else float("inf")

    # ================================================================
    #  Binary File I/O
    # ================================================================

    _BIN_MAGIC = b"MENA"
    _BIN_VERSION = 2

    def save_state(self, filename: str) -> None:
        """
        Save the full crossbar state to a binary .bin file.

        File format:
            - 4 bytes: magic "MENA"
            - 1 byte: version
            - 1 byte: rows
            - 1 byte: cols
            - 1 byte: flags (reserved)
            - 8 bytes: g_min (float64)
            - 8 bytes: g_max (float64)
            - rows × cols × 8 bytes: conductance data (float64)
            - rows × cols × 1 byte: formed flags
            - rows × cols × 8 bytes: cycle counts (int64)
            - rows × cols × 1 byte: stuck-at faults

        Args:
            filename: Output file path.
        """
        with open(filename, "wb") as f:
            f.write(self._BIN_MAGIC)
            f.write(struct.pack("BBBB", self._BIN_VERSION, self.rows, self.cols, 0))
            f.write(struct.pack("dd", self.g_min, self.g_max))
            f.write(self.crossbar.tobytes())
            f.write(self.formed.astype(np.uint8).tobytes())
            f.write(self.cycles.tobytes())
            f.write(self.stuck_at_faults.tobytes())

    def load_state(self, filename: str) -> None:
        """
        Load crossbar state from a .bin file.

        Args:
            filename: Input file path.

        Raises:
            ValueError: If file format is invalid or dimensions don't match.
        """
        with open(filename, "rb") as f:
            magic = f.read(4)
            if magic != self._BIN_MAGIC:
                raise ValueError(f"Invalid file (expected MENA header, got {magic!r})")

            version, rows, cols, flags = struct.unpack("BBBB", f.read(4))
            g_min, g_max = struct.unpack("dd", f.read(16))

            if rows != self.rows or cols != self.cols:
                raise ValueError(
                    f"Dimension mismatch: file is {rows}×{cols}, board is {self.rows}×{self.cols}"
                )

            self.g_min = g_min
            self.g_max = g_max
            self.g_range = g_max - g_min

            n = rows * cols
            self.crossbar = np.frombuffer(f.read(n * 8), dtype=np.float64).reshape(rows, cols).copy()
            self.formed = np.frombuffer(f.read(n), dtype=np.uint8).reshape(rows, cols).astype(bool).copy()
            self.cycles = np.frombuffer(f.read(n * 8), dtype=np.int64).reshape(rows, cols).copy()
            self.stuck_at_faults = np.frombuffer(f.read(n), dtype=np.int8).reshape(rows, cols).copy()

    def export_weights_bin(self, filename: str) -> None:
        """Export the weight matrix (mapped from conductances) as float32 binary."""
        weights = self.conductance_to_weight(self.crossbar)
        weights.astype(np.float32).tofile(filename)

    def import_weights_bin(self, filename: str) -> Dict:
        """
        Import a weight matrix from a float32 binary file and program via VIPI.

        Returns:
            VIPI programming report.
        """
        weights = np.fromfile(filename, dtype=np.float32).reshape(self.rows, self.cols)
        target_g = self.weight_to_conductance(weights)
        return self.vipi_program_matrix(target_g)

    # ================================================================
    #  Linear Algebra Utilities
    # ================================================================

    def eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues of the conductance matrix (if square)."""
        if self.rows != self.cols:
            raise ValueError("Eigenvalues require a square crossbar")
        return np.linalg.eigvals(self.crossbar)

    def singular_values(self) -> np.ndarray:
        """Compute singular values of the conductance matrix (SVD)."""
        return np.linalg.svd(self.crossbar, compute_uv=False)

    def condition_number(self) -> float:
        """Condition number of the conductance matrix (ratio of largest to smallest singular value)."""
        sv = self.singular_values()
        return float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")

    def frobenius_norm(self) -> float:
        """Frobenius norm of the conductance matrix: ||G||_F = sqrt(Σ G(i,j)²)."""
        return float(np.linalg.norm(self.crossbar, "fro"))

    def rank(self, tol: Optional[float] = None) -> int:
        """Numerical rank of the conductance matrix."""
        return int(np.linalg.matrix_rank(self.crossbar, tol=tol))

    def determinant(self) -> float:
        """Determinant of the conductance matrix (square only)."""
        if self.rows != self.cols:
            raise ValueError("Determinant requires a square crossbar")
        return float(np.linalg.det(self.crossbar))

    def pseudo_inverse(self) -> np.ndarray:
        """Moore-Penrose pseudo-inverse of the conductance matrix."""
        return np.linalg.pinv(self.crossbar)

    # ================================================================
    #  Internals
    # ================================================================

    def _is_stuck(self, row: int, col: int) -> bool:
        return self.stuck_at_faults[row, col] != 0

    def _log(self, op: str, row: int, col: int, g: float) -> None:
        self.programming_log.append({"op": op, "row": row, "col": col, "g": g})

    def __repr__(self) -> str:
        return (
            f"OpenMenaBoard(rows={self.rows}, cols={self.cols}, "
            f"G=[{self.g_min:.1e}, {self.g_max:.1e}] S, "
            f"utilization={self.crossbar_utilization():.1f}%, "
            f"yield={self.programming_yield():.1f}%)"
        )


# ====================================================================
#  Demo
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  OpenMENA API Demo — Digit Recognition Workflow")
    print("=" * 60)

    # 1. Create board (64 inputs × 10 outputs for digit classification)
    board = OpenMenaBoard(rows=64, cols=10, g_min=1e-6, g_max=1e-3, seed=42)
    print(f"\n[1] Board initialized: {board}")

    # 2. Inject manufacturing defects
    n_faults = board.apply_stuck_at_faults(fault_rate=0.01)
    print(f"\n[2] Injected {n_faults} stuck-at faults")
    print(f"    Programming yield: {board.programming_yield():.1f}%")

    # 3. Generate mock pre-trained weights and transfer via VIPI
    np.random.seed(42)
    mock_weights = np.random.uniform(-1, 1, size=(64, 10))
    print(f"\n[3] Transferring weights via VIPI algorithm...")
    report = board.weight_transfer(mock_weights)
    print(f"    VIPI yield: {report['yield_pct']:.1f}%")
    print(f"    Total programming cycles: {report['total_cycles']:,}")
    print(f"    Failed cells: {len(report['failed_cells'])}")

    # 4. Run inference on a mock 8×8 image (flattened to 64)
    mock_input = np.random.uniform(0, 1, size=64)
    logits = board.matrix_vector_multiply(mock_input)
    probs = board.softmax(logits)
    prediction = int(np.argmax(probs))
    print(f"\n[4] Inference result:")
    print(f"    Predicted class: {prediction}")
    print(f"    Confidence: {probs[prediction]:.4f}")

    # 5. Hardware metrics
    print(f"\n[5] Hardware Metrics:")
    print(f"    Energy/MAC: {board.energy_per_mac():.2e} J")
    print(f"    Crossbar utilization: {board.crossbar_utilization():.1f}%")
    print(f"    SNR: {board.snr():.1f} dB")
    print(f"    Throughput @ 100 MHz: {board.throughput_tops():.4f} TOPS")
    print(f"    Power (1V input): {board.power_consumption(np.ones(64)):.4e} W")

    # 6. Linear algebra
    print(f"\n[6] Matrix properties:")
    print(f"    Frobenius norm: {board.frobenius_norm():.4e}")
    print(f"    Rank: {board.rank()}")

    # 7. Save and reload state
    board.save_state("openmena_demo_state.bin")
    print(f"\n[7] State saved to openmena_demo_state.bin")

    board2 = OpenMenaBoard(rows=64, cols=10, seed=99)
    board2.load_state("openmena_demo_state.bin")
    diff = np.max(np.abs(board.crossbar - board2.crossbar))
    print(f"    Reload verification — max diff: {diff:.1e}")

    print(f"\n{'=' * 60}")
    print(f"  Demo complete.")
    print(f"{'=' * 60}")
