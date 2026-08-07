import sys
import os
import argparse
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.optim as optim

from src.domain import BurgersDomain
from src.qapinn import HybridQAPINN, compute_qapinn_pde_residual, count_parameters
from src.quantum_diagnostics import QuantumDiagnostics


# =============================================================================
# INTERACTIVE QUBIT CONFIGURATION PROMPT
# =============================================================================

def ask_qubit_count():
    """
    Interactively asks the user how many qubits to use and validates the input.
    Supported values: 3, 4, 5.  Default is 4.
    """
    print()
    print("+====================================================================+")
    print("|       HYBRID QAPINN  --  WISER Summer Program 2026                |")
    print("|       Burgers Equation: u_t + u*u_x = nu*u_xx  (nu = 0.01)       |")
    print("+====================================================================+")
    print()
    print("  Supported qubit counts  :  3, 4, or 5")
    print("  Recommended (default)   :  4 qubits")
    print()

    while True:
        raw = input("  >>  How many qubits do you want to use? "
                    "[3 / 4 / 5, press Enter for 4]: ").strip()

        if raw == "":
            n_qubits = 4
            print("  Using default: 4 qubits")
            break
        if raw in ("3", "4", "5"):
            n_qubits = int(raw)
            print(f"  Selected: {n_qubits} qubits")
            break
        else:
            print(f"  [!] Invalid input '{raw}'.  Please enter 3, 4, or 5.")

    print()
    return n_qubits


# =============================================================================
# INTERACTIVE BACKEND SELECTION PROMPT
# =============================================================================

BACKEND_OPTIONS = {
    '1': 'default.qubit',
    '2': 'lightning.qubit',
    '3': 'lightning.gpu',
}

BACKEND_INFO = {
    'default.qubit'  : ('Always available  — pure Python/NumPy simulator',
                        'backprop',  '1x  (baseline)',  'None'),
    'lightning.qubit': ('Requires: pip install pennylane-lightning',
                        'adjoint',   '5-10x faster',   'None (CPU only)'),
    'lightning.gpu'  : ('Requires: CUDA + pip install pennylane-lightning[gpu]',
                        'adjoint',   '20-50x faster',  'NVIDIA GPU + CUDA 11/12'),
}

def ask_backend() -> str:
    """
    Interactively asks the user which PennyLane backend device to use.
    Returns the device name string.
    """
    print("+====================================================================+")
    print("|  SELECT QUANTUM BACKEND                                            |")
    print("+====================================================================+")
    print()
    print("  [1]  default.qubit    -- Always available, pure Python  (baseline speed)")
    print("  [2]  lightning.qubit  -- C++ CPU backend,  5-10x faster (recommended)")
    print("  [3]  lightning.gpu    -- NVIDIA GPU,       20-50x faster (needs CUDA)")
    print()
    print("  Speed comparison for 5000 epochs on this machine (approx):")
    print("    default.qubit   :  ~1-2 hours")
    print("    lightning.qubit :  ~10-20 minutes")
    print("    lightning.gpu   :  ~2-5 minutes  (if NVIDIA GPU available)")
    print()

    while True:
        raw = input("  >>  Choose backend [1 / 2 / 3, press Enter for 2]: ").strip()
        if raw == "":
            chosen = 'lightning.qubit'
            print("  Using default: lightning.qubit  (5-10x faster, no GPU needed)")
            break
        if raw in BACKEND_OPTIONS:
            chosen = BACKEND_OPTIONS[raw]
            print(f"  Selected: {chosen}")
            break
        else:
            print(f"  [!] Invalid input '{raw}'. Please enter 1, 2, or 3.")

    # Show install guide if not default
    if chosen != 'default.qubit':
        desc, diff, speed, req = BACKEND_INFO[chosen]
        print()
        print(f"  Backend      : {chosen}")
        print(f"  Diff method  : {diff}  (auto-selected)")
        print(f"  Speed gain   : {speed}")
        print(f"  Requirements : {req}")
        print(f"  Install note : {desc}")

    print()
    return chosen



# =============================================================================
# VISUAL CIRCUIT DIAGRAM  (Qiskit / IBM Quantum Composer style via matplotlib)
# =============================================================================

def print_circuit_diagram(n_qubits: int, n_layers: int = 2):
    """
    Renders and DISPLAYS a Qiskit / IBM Quantum Composer-style visual circuit
    diagram using matplotlib.  The image is also saved to:
        evaluation/plots/vqc_circuit_{n_qubits}qubit.png

    Color coding:
      Blue  = Encoding gates  (RX/RY, fixed from x,t)
      Green = Variational rotation gates  (RX, RY, RZ -- learnable weights)
      Red   = CNOT entanglement gates     (fixed structure)
      Gray  = Pauli-Z measurement
    """
    import matplotlib
    matplotlib.use('TkAgg')          # interactive window backend on Windows
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle

    # ── layout constants ──────────────────────────────────────────────────────
    FIG_W      = 5 + n_qubits * 1.5 + n_layers * 3.5   # figure width scales with circuit size
    FIG_H      = max(5.0, n_qubits * 1.2 + 2.0)
    WIRE_START = 0.9
    WIRE_END   = FIG_W - 0.6

    # y positions: qubit 0 at top
    y_pos = [FIG_H - 1.4 - q * 1.2 for q in range(n_qubits)]

    # x column positions
    ENC_X    = 1.8                                    # encoding gate column
    VAR_STEP = 0.95                                   # spacing between RX/RY/RZ
    L1_START = ENC_X + 1.3                           # variational layer 1 start
    L1_END   = L1_START + (3 * VAR_STEP)             # after 3 gates
    CNOT1_X  = L1_END + 0.45                         # CNOT column after layer 1
    L2_START = CNOT1_X + 0.80                        # variational layer 2 start
    L2_END   = L2_START + (3 * VAR_STEP)
    CNOT2_X  = L2_END + 0.45
    MEAS_X   = CNOT2_X + 0.80

    # ── colors ─────────────────────────────────────────────────────────────────
    BG      = '#0d1117'
    WIRE_C  = '#c9d1d9'
    BLUE    = '#388bfd'    # encoding
    GREEN   = '#3fb950'    # variational
    RED     = '#f85149'    # CNOT
    GRAY    = '#8b949e'    # measurement
    TEXT_C  = '#ffffff'
    TITLE_C = '#58a6ff'
    PARAM_C = '#e3b341'
    BOX_BG  = '#161b22'

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')

    # ── helper functions ───────────────────────────────────────────────────────
    def gate(x, y, label, color, fw=0.78, fh=0.42, fs=7.8):
        """Draw a rounded gate box with label."""
        box = FancyBboxPatch(
            (x - fw / 2, y - fh / 2), fw, fh,
            boxstyle='round,pad=0.04', linewidth=1.3,
            edgecolor='white', facecolor=color, zorder=4
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fs, color=TEXT_C, fontweight='bold',
                fontfamily='DejaVu Sans Mono', zorder=5)

    def cnot_gate(x, y_ctrl, y_tgt):
        """Draw a CNOT gate: filled dot on control, circle+cross on target."""
        ax.plot([x, x], [y_ctrl, y_tgt], color=RED, lw=1.8, zorder=3)
        ax.add_patch(Circle((x, y_ctrl), 0.10, color=RED, zorder=5))
        ax.add_patch(Circle((x, y_tgt), 0.20,
                             facecolor=BOX_BG, edgecolor=RED, lw=1.8, zorder=4))
        ax.plot([x - 0.20, x + 0.20], [y_tgt, y_tgt], color=RED, lw=1.6, zorder=5)
        ax.plot([x, x], [y_tgt - 0.20, y_tgt + 0.20], color=RED, lw=1.6, zorder=5)

    def meas_box(x, y):
        """Draw a measurement box (gauge meter icon)."""
        bx = FancyBboxPatch(
            (x - 0.44, y - 0.26), 0.88, 0.52,
            boxstyle='round,pad=0.04', linewidth=1.3,
            edgecolor='white', facecolor=GRAY, zorder=4
        )
        ax.add_patch(bx)
        # arc
        import numpy as np
        th = np.linspace(np.pi, 0, 40)
        ax.plot(x + 0.16 * np.cos(th), y - 0.04 + 0.14 * np.sin(th),
                color='white', lw=1.2, zorder=5)
        # needle
        ax.annotate('', xy=(x + 0.11, y + 0.06), xytext=(x, y - 0.04),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.2), zorder=5)
        ax.text(x, y - 0.38, r'$\langle Z \rangle$', ha='center', va='top',
                fontsize=7, color=PARAM_C, zorder=5)

    # ── stage background bands ─────────────────────────────────────────────────
    band_alpha = 0.06
    bands = [
        (WIRE_START, ENC_X + 0.7,   '#388bfd', 'ENCODING\n(fixed)'),
        (ENC_X + 0.7, CNOT1_X + 0.5, '#3fb950', 'VARIATIONAL LAYER 1\n(learnable)'),
        (CNOT1_X + 0.5, CNOT2_X + 0.5, '#3fb950', 'VARIATIONAL LAYER 2\n(learnable)'),
        (CNOT2_X + 0.5, MEAS_X + 0.6,  '#8b949e', 'MEASURE'),
    ]
    for (x0, x1, col, lbl) in bands:
        ax.add_patch(plt.Rectangle(
            (x0, 0.3), x1 - x0, FIG_H - 0.9,
            facecolor=col, alpha=band_alpha, zorder=0
        ))
        ax.text((x0 + x1) / 2, FIG_H - 0.55, lbl, ha='center', va='center',
                fontsize=7, color=col, fontweight='bold', zorder=6,
                multialignment='center')

    # ── draw qubit wires and labels ────────────────────────────────────────────
    for q, y in enumerate(y_pos):
        ax.plot([WIRE_START, WIRE_END], [y, y], color=WIRE_C, lw=1.5, zorder=1)
        ax.text(0.5, y, f'$q_{q}$', ha='center', va='center',
                fontsize=11, color=WIRE_C, fontweight='bold', zorder=6)

    # ── ENCODING GATES ─────────────────────────────────────────────────────────
    enc_labels = ['RX(x·π)' if q % 2 == 0 else 'RY(t·π)' for q in range(n_qubits)]
    for q, (y, lbl) in enumerate(zip(y_pos, enc_labels)):
        gate(ENC_X, y, lbl, BLUE, fw=0.90, fs=7.5)

    # ── VARIATIONAL LAYER 1 ────────────────────────────────────────────────────
    rot_lbls = ['RX(w)', 'RY(w)', 'RZ(w)']
    for ri, rl in enumerate(rot_lbls):
        x = L1_START + ri * VAR_STEP
        for y in y_pos:
            gate(x, y, rl, GREEN, fw=0.76, fs=7.5)

    for q in range(n_qubits - 1):
        cnot_gate(CNOT1_X, y_pos[q], y_pos[q + 1])

    # ── VARIATIONAL LAYER 2 ────────────────────────────────────────────────────
    for ri, rl in enumerate(rot_lbls):
        x = L2_START + ri * VAR_STEP
        for y in y_pos:
            gate(x, y, rl, GREEN, fw=0.76, fs=7.5)

    for q in range(n_qubits - 1):
        cnot_gate(CNOT2_X, y_pos[q], y_pos[q + 1])

    # ── MEASUREMENT ────────────────────────────────────────────────────────────
    for y in y_pos:
        meas_box(MEAS_X, y)

    # ── TITLE ──────────────────────────────────────────────────────────────────
    ax.text(FIG_W / 2, FIG_H - 0.20,
            f'VQC for QA-PINN  —  {n_qubits} Qubits, {n_layers} Variational Layers',
            ha='center', va='center', fontsize=13, color=TITLE_C, fontweight='bold')

    # ── FOOTER (parameter count) ────────────────────────────────────────────────
    q_p    = n_qubits * 3 * n_layers
    fc1    = n_qubits * 40 + 40
    total  = q_p + fc1 + 40 * 40 + 40 + 41
    red_pct = (3441 - total) / 3441 * 100
    footer = (f'{q_p} quantum params  |  Linear CNOT entanglement  |  '
              f'Angle encoding  |  Total: {total} params  |  {red_pct:.1f}% reduction vs c-PINN')
    ax.text(FIG_W / 2, 0.15, footer,
            ha='center', va='center', fontsize=8, color=PARAM_C)

    # ── LEGEND ─────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=BLUE,  edgecolor='white', label='Encoding  (not learnable)'),
        mpatches.Patch(facecolor=GREEN, edgecolor='white', label='Variational RX/RY/RZ  (learnable)'),
        mpatches.Patch(facecolor=RED,   edgecolor='white', label='CNOT Entanglement  (fixed)'),
        mpatches.Patch(facecolor=GRAY,  edgecolor='white', label='Pauli-Z Measurement'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
              facecolor=BOX_BG, edgecolor='#30363d', labelcolor='white',
              framealpha=0.92, ncol=2,
              bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout(pad=0.4)

    # ── Save to file ────────────────────────────────────────────────────────────
    os.makedirs('evaluation/plots', exist_ok=True)
    save_path = f'evaluation/plots/vqc_circuit_{n_qubits}qubit.png'
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  [Circuit] Diagram saved to: {save_path}")
    print("  [Circuit] Displaying circuit window ...  (close it to continue)")

    plt.show()     # opens interactive window; training starts after user closes it
    plt.close(fig)


# =============================================================================
# SINGLE-MODEL TRAINING FUNCTION
# =============================================================================

def train_qapinn_model(n_qubits=4, position='first', measurement_mode='expectation',
                        encoding='angle', epochs=5000, lr=1e-3, nu=0.01,
                        seed=42, save_name='qapinn_primary',
                        device_name='default.qubit'):
    """
    Trains a single Hybrid QAPINN model and logs accuracy,
    parameter compression, and quantum diagnostics.
    """
    print(f"\n==================================================================")
    print(f"Starting QAPINN Training: {save_name}")
    print(f"  Config: {n_qubits} Qubits | Position: '{position}' | "
          f"Measurement: '{measurement_mode}'")
    print(f"  Backend: {device_name}")
    print(f"  Target Epochs: {epochs}")
    print(f"==================================================================")

    # 1. Domain & Training Data
    domain = BurgersDomain(seed=seed)
    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data(
        N_f=2000, N_i=200, N_b=200)
    X_mesh, T_mesh, X_eval, x_flat, t_flat = domain.generate_eval_grid(
        Nx=256, Nt=100)

    u_exact = np.load('evaluation/exact_solution.npy')

    # 2. Model, Diagnostics, Optimizer
    model = HybridQAPINN(
        n_qubits=n_qubits,
        position=position,
        measurement_mode=measurement_mode,
        encoding=encoding,
        device_name=device_name
    )
    param_info = count_parameters(model)
    print(f"  Parameters: Total={param_info['total']}, "
          f"Quantum={param_info['quantum']}, "
          f"Classical={param_info['classical']}, "
          f"Compression={param_info['reduction_pct']:.2f}%")

    diagnostics = QuantumDiagnostics(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Metrics History
    history = {
        'epoch': [], 'loss_total': [], 'loss_f': [],
        'loss_i': [], 'loss_b': [], 'relative_l2': [],
        'von_neumann_entropy': [], 'gradient_variance': []
    }

    start_time = time.time()
    weights = (1.0, 10.0, 10.0)

    save_dir = os.path.join('evaluation', save_name)
    os.makedirs(save_dir, exist_ok=True)

    # Epoch-0 XAI snapshot
    act_0 = diagnostics.capture_post_quantum_activations(X_eval)
    np.save(os.path.join(save_dir, 'act_epoch_0.npy'), act_0)

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        residual, _ = compute_qapinn_pde_residual(model, x_f, t_f, nu=nu)
        loss_f = torch.mean(residual ** 2)

        u_i_pred = model(X_i)
        loss_i = torch.mean((u_i_pred - u_i) ** 2)

        u_b_pred = model(X_b)
        loss_b = torch.mean((u_b_pred - u_b) ** 2)

        total_loss = weights[0] * loss_f + weights[1] * loss_i + weights[2] * loss_b
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Log every 100 epochs
        if epoch % 100 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                u_pred = model(X_eval).cpu().numpy().reshape(u_exact.shape)
                l2_err = float(
                    np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))

            # Quantum diagnostics every 500 epochs
            if epoch % 500 == 0 or epoch == epochs:
                entropy = diagnostics.compute_von_neumann_entropy(X_eval[:50])

                def eval_loss_fn():
                    res, _ = compute_qapinn_pde_residual(
                        model, x_f[:100], t_f[:100], nu=nu)
                    return torch.mean(res ** 2)

                var_grad, _ = diagnostics.compute_quantum_gradient_variance(
                    eval_loss_fn)
            else:
                entropy  = (history['von_neumann_entropy'][-1]
                            if history['von_neumann_entropy'] else 0.0)
                var_grad = (history['gradient_variance'][-1]
                            if history['gradient_variance'] else 0.0)

            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['loss_f'].append(loss_f.item())
            history['loss_i'].append(loss_i.item())
            history['loss_b'].append(loss_b.item())
            history['relative_l2'].append(l2_err)
            history['von_neumann_entropy'].append(entropy)
            history['gradient_variance'].append(var_grad)

            if epoch % 500 == 0 or epoch == epochs:
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Loss: {total_loss.item():.5e} | "
                      f"Rel L2: {l2_err:.4e} | "
                      f"Entropy S(rho): {entropy:.4f} | "
                      f"Grad Var: {var_grad:.4e}")

        # XAI snapshots at key checkpoints
        if epoch in [1000, 5000]:
            act = diagnostics.capture_post_quantum_activations(X_eval)
            np.save(os.path.join(save_dir, f'act_epoch_{epoch}.npy'), act)

    elapsed = time.time() - start_time
    print("==================================================================")
    print(f"Training Complete for {save_name} in {elapsed:.2f} seconds!")
    print(f"Final Relative L2 Error: {history['relative_l2'][-1]:.4e}")
    print("==================================================================")

    # 5. Save Outputs
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    np.save(os.path.join(save_dir, 'history.npy'), history)
    np.save(os.path.join(save_dir, 'param_info.npy'), param_info)

    model.eval()
    with torch.no_grad():
        u_final = model(X_eval).cpu().numpy().reshape(u_exact.shape)
    np.save(os.path.join(save_dir, 'predictions.npy'), u_final)

    return model, history, param_info


# =============================================================================
# BENCHMARK SUITE  (optional --all flag)
# =============================================================================

def run_all_benchmarks(epochs=5000):
    """Runs a full benchmark suite across 6 QAPINN configurations."""
    print("\n>>> Running Full Benchmark Suite across 6 QAPINN Configurations <<<")

    benchmarks = {}

    cfg = [
        dict(n_qubits=4, position='first',  measurement_mode='expectation', save_name='qapinn_primary'),
        dict(n_qubits=3, position='first',  measurement_mode='expectation', save_name='qapinn_3qubit'),
        dict(n_qubits=5, position='first',  measurement_mode='expectation', save_name='qapinn_5qubit'),
        dict(n_qubits=4, position='middle', measurement_mode='expectation', save_name='qapinn_pos2'),
        dict(n_qubits=4, position='final',  measurement_mode='expectation', save_name='qapinn_pos3'),
        dict(n_qubits=4, position='first',  measurement_mode='probability', save_name='qapinn_probability'),
    ]

    for c in cfg:
        _, h, info = train_qapinn_model(epochs=epochs, **c)
        benchmarks[c['save_name']] = (h, info)

    print("\n==================================================================")
    print("ALL QAPINN BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("==================================================================")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train Hybrid QAPINN for WISER Program 2026")
    parser.add_argument(
        '--all', action='store_true',
        help="Run full benchmark suite across all 6 configurations")
    parser.add_argument(
        '--epochs', type=int, default=5000,
        help="Number of training epochs (default: 5000)")
    parser.add_argument(
        '--qubits', type=int, default=None,
        help="Number of qubits (3, 4, or 5). "
             "If omitted, you will be asked interactively.")
    parser.add_argument(
        '--backend', type=str, default=None,
        choices=['default.qubit', 'lightning.qubit', 'lightning.gpu'],
        help="PennyLane backend device. If omitted, you will be asked interactively.")
    args = parser.parse_args()

    if args.all:
        # ── Benchmark mode: skip interactive prompts ────────────────────
        run_all_benchmarks(epochs=args.epochs)

    else:
        # ── Step 1: Ask user for qubit count ──────────────────────────────
        if args.qubits is not None:
            if args.qubits not in (3, 4, 5):
                print(f"  [!] --qubits must be 3, 4, or 5. "
                      f"Got {args.qubits}. Defaulting to 4.")
                n_qubits = 4
            else:
                n_qubits = args.qubits
                print(f"  Using --qubits flag: {n_qubits} qubits")
        else:
            n_qubits = ask_qubit_count()

        # ── Step 2: Print the VQC circuit diagram ─────────────────────────
        print_circuit_diagram(n_qubits=n_qubits, n_layers=2)

        # ── Step 3: Confirm before training starts ────────────────────────
        print("  Press Enter to begin training, or Ctrl+C to cancel ...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n  Training cancelled.")
            sys.exit(0)

        # ── Step 4: Train ──────────────────────────────────────────────────
        save_name = f'qapinn_{n_qubits}qubit'
        train_qapinn_model(
            n_qubits=n_qubits,
            position='first',
            measurement_mode='expectation',
            epochs=args.epochs,
            save_name=save_name
        )
