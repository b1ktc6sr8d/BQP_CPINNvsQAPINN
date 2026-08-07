"""
barren_plateau_scaling.py
==========================
WISER Summer Program 2026 — Part 2 Quantum Analysis

Experiment:
    Sweep qubit count N from 2 to 6.
    For each N, train a hybrid QAPINN for 500 epochs and track the
    variance of all quantum rotation-parameter gradients at every epoch.
    Finally, plot Qubit Count vs Mean Gradient Variance on a log scale,
    overlaying the Barren Plateau threshold line at 1e-7.

Theory:
    In a random quantum circuit, gradient variance scales as:
        Var(dL/dθ_k) ∝ 1 / 2^N
    So variance drops EXPONENTIALLY as N grows.  When Var < 1e-7 the
    optimizer receives no meaningful signal → training stalls (Barren Plateau).

Run:
    python barren_plateau_scaling.py
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')         # interactive window -- pops up automatically
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pennylane as qml

# ── ensure src/ is on the path ────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.domain import BurgersDomain
from src.qapinn import HybridQAPINN, compute_qapinn_pde_residual


# =============================================================================
# CONFIGURATION
# =============================================================================

QUBIT_RANGE   = list(range(2, 7))   # N = 2, 3, 4, 5, 6
N_EPOCHS      = 500                  # epochs per qubit configuration
N_F           = 500                  # PDE collocation points  (small for speed)
N_I           = 100                  # initial condition points
N_B           = 100                  # boundary condition points
LR            = 1e-3                 # Adam learning rate
NU            = 0.01                 # viscosity coefficient
SEED          = 42
BARREN_THRESH = 1e-7                 # Barren Plateau warning threshold
DEVICE_NAME   = 'lightning.qubit'   # CPU C++ backend (5-10x faster; no GPU needed)

SAVE_PATH = 'evaluation/plots/barren_plateau_scaling.png'
os.makedirs('evaluation/plots', exist_ok=True)


# =============================================================================
# GRADIENT VARIANCE TRACKER
# =============================================================================

def get_quantum_grad_variance(model: HybridQAPINN) -> float:
    """
    Collect the gradients of all quantum rotation parameters (qlayer weights)
    and return their variance.

    Returns 0.0 if no gradients exist yet (before first backward pass).
    """
    grads = []
    for name, param in model.named_parameters():
        if 'qlayer' in name and param.grad is not None:
            grads.append(param.grad.detach().cpu().flatten())

    if len(grads) == 0:
        return 0.0

    all_grads = torch.cat(grads)        # shape: (total_quantum_params,)
    return float(all_grads.var().item())


# =============================================================================
# SINGLE QUBIT-COUNT EXPERIMENT
# =============================================================================

def run_experiment(n_qubits: int,
                   domain: BurgersDomain,
                   u_exact: np.ndarray,
                   device_name: str = 'lightning.qubit') -> list:
    """
    Train a QAPINN with n_qubits qubits for N_EPOCHS epochs.
    At every epoch, record the quantum parameter gradient variance.

    Returns:
        List of float — gradient variance at each epoch (length = N_EPOCHS).
    """
    print(f"\n  [Qubits={n_qubits}] Initialising QAPINN ...")

    # ── Data ─────────────────────────────────────────────────────────────────
    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data(
        N_f=N_F, N_i=N_I, N_b=N_B)
    _, _, X_eval, _, _ = domain.generate_eval_grid(Nx=128, Nt=50)

    # ── Model & Optimizer ─────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    model = HybridQAPINN(
        n_qubits=n_qubits,
        n_layers=2,
        position='first',
        measurement_mode='expectation',
        encoding='angle',
        entanglement='linear',
        device_name=device_name
    )
    q_param_count = sum(
        p.numel() for name, p in model.named_parameters()
        if 'qlayer' in name
    )
    print(f"  [Qubits={n_qubits}] Quantum params: {q_param_count}  "
          f"| Total params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_weights = (1.0, 10.0, 10.0)

    variance_history = []

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # PDE residual loss
        residual, _ = compute_qapinn_pde_residual(model, x_f, t_f, nu=NU)
        loss_f = torch.mean(residual ** 2)

        # Initial condition loss
        u_i_pred = model(X_i)
        loss_i = torch.mean((u_i_pred - u_i) ** 2)

        # Boundary condition loss
        u_b_pred = model(X_b)
        loss_b = torch.mean((u_b_pred - u_b) ** 2)

        total_loss = (loss_weights[0] * loss_f
                      + loss_weights[1] * loss_i
                      + loss_weights[2] * loss_b)
        total_loss.backward()

        # ── Record gradient variance BEFORE optimizer.step() ─────────────────
        var_now = get_quantum_grad_variance(model)
        variance_history.append(var_now)

        optimizer.step()

        if epoch % 100 == 0 or epoch == N_EPOCHS:
            print(f"    Epoch {epoch:4d}/{N_EPOCHS} | "
                  f"Loss: {total_loss.item():.4e} | "
                  f"Grad Var: {var_now:.3e}")

    mean_var = float(np.mean(variance_history))
    status   = "SAFE" if mean_var > BARREN_THRESH else "BARREN PLATEAU!"
    print(f"  [Qubits={n_qubits}] Mean grad variance: {mean_var:.3e}  --> {status}")

    return variance_history


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def main():
    print("=" * 68)
    print("  BARREN PLATEAU SCALING EXPERIMENT")
    print(f"  Qubit range : N = {QUBIT_RANGE[0]} to {QUBIT_RANGE[-1]}")
    print(f"  Epochs each : {N_EPOCHS}")
    print(f"  Backend     : {DEVICE_NAME}")
    print(f"  Threshold   : {BARREN_THRESH:.0e}  (Barren Plateau boundary)")
    print("=" * 68)

    # Shared domain & exact solution (generated once for all N)
    domain  = BurgersDomain(seed=SEED)
    u_exact = np.load('evaluation/exact_solution.npy')

    # Results containers
    all_histories  = {}     # n_qubits -> [var per epoch]
    mean_variances = {}     # n_qubits -> scalar mean variance

    for n_qubits in QUBIT_RANGE:
        hist = run_experiment(n_qubits, domain, u_exact, device_name=DEVICE_NAME)
        all_histories[n_qubits]  = hist
        mean_variances[n_qubits] = float(np.mean(hist))

    # ── Save raw data ─────────────────────────────────────────────────────────
    np.save('evaluation/barren_plateau_histories.npy', all_histories)
    print("\n  Raw variance histories saved to "
          "evaluation/barren_plateau_histories.npy")

    # =========================================================================
    # PLOTTING
    # =========================================================================

    # ── Color palette ─────────────────────────────────────────────────────────
    BG      = '#0d1117'
    AX_BG   = '#161b22'
    GRID_C  = '#30363d'
    WIRE_C  = '#c9d1d9'
    TITLE_C = '#58a6ff'
    PARAM_C = '#e3b341'
    RED_C   = '#f85149'
    GREEN_C = '#3fb950'

    qubit_vals = list(QUBIT_RANGE)
    var_vals   = [mean_variances[n] for n in qubit_vals]

    # Safe vs. plateau color coding per point
    point_colors = [GREEN_C if v > BARREN_THRESH else RED_C for v in var_vals]

    fig, (ax_main, ax_trace) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={'width_ratios': [1, 1.4]}
    )
    fig.patch.set_facecolor(BG)

    # ── LEFT: Qubit Count vs Mean Gradient Variance ────────────────────────
    ax_main.set_facecolor(AX_BG)
    ax_main.set_yscale('log')

    # Barren Plateau threshold band
    ax_main.axhspan(0, BARREN_THRESH,
                    color=RED_C, alpha=0.12, label='Barren Plateau Zone')
    ax_main.axhline(BARREN_THRESH,
                    color=RED_C, lw=1.8, ls='--',
                    label=f'Threshold  1×10⁻⁷')

    # Theoretical exponential decay reference: Var ∝ 1/2^N
    ref_N   = np.linspace(2, 6, 100)
    ref_var = var_vals[0] * (2 ** qubit_vals[0]) / (2 ** ref_N)
    ax_main.plot(ref_N, ref_var,
                 color=PARAM_C, lw=1.2, ls=':',
                 label=r'Theory: Var $\propto 1/2^N$', alpha=0.8)

    # Actual measurements
    ax_main.plot(qubit_vals, var_vals,
                 color=WIRE_C, lw=2.0, zorder=3, alpha=0.8)
    sc = ax_main.scatter(qubit_vals, var_vals,
                         c=point_colors, s=130, zorder=5,
                         edgecolors='white', linewidths=1.2,
                         label='Measured Var(∂L/∂θ)')

    # Annotate each point
    for n, v in zip(qubit_vals, var_vals):
        status = 'SAFE' if v > BARREN_THRESH else 'BP!'
        col    = GREEN_C if v > BARREN_THRESH else RED_C
        ax_main.annotate(
            f'N={n}\n{v:.1e}\n{status}',
            xy=(n, v), xytext=(n + 0.08, v * 1.6),
            fontsize=7.5, color=col, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=col, lw=0.8)
        )

    ax_main.set_xlabel('Number of Qubits (N)',
                        fontsize=12, color=WIRE_C)
    ax_main.set_ylabel('Mean Gradient Variance  Var(∂L/∂θ)  [log scale]',
                        fontsize=11, color=WIRE_C)
    ax_main.set_title('Qubit Count vs Quantum Gradient Variance\n'
                       '(Barren Plateau Analysis)',
                       fontsize=13, color=TITLE_C, fontweight='bold', pad=12)
    ax_main.set_xticks(qubit_vals)
    ax_main.tick_params(colors=WIRE_C)
    ax_main.spines[:].set_color(GRID_C)
    ax_main.grid(True, color=GRID_C, lw=0.7, alpha=0.7, which='both')
    ax_main.yaxis.set_minor_locator(ticker.LogLocator(subs='all'))
    ax_main.legend(fontsize=8.5, facecolor=BG, edgecolor=GRID_C,
                   labelcolor=WIRE_C, loc='upper right')

    # ── RIGHT: Per-epoch variance traces for each qubit count ─────────────
    ax_trace.set_facecolor(AX_BG)
    ax_trace.set_yscale('log')

    cmap   = plt.cm.get_cmap('plasma', len(QUBIT_RANGE))
    epochs = list(range(1, N_EPOCHS + 1))

    for idx, n in enumerate(QUBIT_RANGE):
        hist = all_histories[n]
        # Replace exact zeros (before first gradient) with tiny value for log scale
        hist_safe = [max(v, 1e-14) for v in hist]
        ax_trace.plot(epochs, hist_safe,
                      color=cmap(idx), lw=1.4, alpha=0.85,
                      label=f'N={n}  (mean={mean_variances[n]:.1e})')

    ax_trace.axhline(BARREN_THRESH,
                     color=RED_C, lw=2.0, ls='--',
                     label=f'Barren Plateau threshold  1×10⁻⁷')
    ax_trace.axhspan(0, BARREN_THRESH, color=RED_C, alpha=0.10)

    ax_trace.set_xlabel('Training Epoch',
                         fontsize=12, color=WIRE_C)
    ax_trace.set_ylabel('Gradient Variance  Var(∂L/∂θ)  [log scale]',
                         fontsize=11, color=WIRE_C)
    ax_trace.set_title('Gradient Variance Trajectory per Qubit Count\n'
                        '(over 500 epochs)',
                        fontsize=13, color=TITLE_C, fontweight='bold', pad=12)
    ax_trace.tick_params(colors=WIRE_C)
    ax_trace.spines[:].set_color(GRID_C)
    ax_trace.grid(True, color=GRID_C, lw=0.7, alpha=0.7, which='both')
    ax_trace.yaxis.set_minor_locator(ticker.LogLocator(subs='all'))
    ax_trace.legend(fontsize=8.5, facecolor=BG, edgecolor=GRID_C,
                    labelcolor=WIRE_C, loc='upper right')

    # ── Footer ────────────────────────────────────────────────────────────
    footer = (
        'WISER Summer Program 2026  |  QA-PINN Barren Plateau Experiment  |  '
        'VQC: linear CNOT entanglement, angle encoding, 2 variational layers'
    )
    fig.text(0.5, 0.01, footer,
             ha='center', va='bottom', fontsize=8.5,
             color=PARAM_C)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(SAVE_PATH, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved to: {SAVE_PATH}")
    print("  Displaying plot window ... (close it to exit)")
    plt.show()     # <-- pops up the image window automatically
    plt.close(fig)

    # =========================================================================
    # CONSOLE RESULTS TABLE
    # =========================================================================

    print()
    print("=" * 60)
    print(f"  {'Qubits':>8}  {'Quantum Params':>16}  "
          f"{'Mean Grad Var':>14}  {'Status':>12}")
    print("  " + "-" * 56)
    for n in QUBIT_RANGE:
        q_params = n * 3 * 2         # n_qubits * 3 rotations * 2 layers
        mv       = mean_variances[n]
        status   = "SAFE (active)" if mv > BARREN_THRESH else "BARREN PLATEAU"
        print(f"  {n:>8}  {q_params:>16}  {mv:>14.3e}  {status:>12}")
    print("=" * 60)
    print()
    print("  Theoretical prediction: Var(dL/dθ) ∝ 1 / 2^N")
    print(f"  Barren Plateau threshold: {BARREN_THRESH:.0e}")
    print()
    print(f"  Plot saved  →  {SAVE_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
