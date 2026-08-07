"""
train_heat_equation.py
======================
WISER Summer Program 2026 — Linear PDE Experiment

Experiment:
    Replace the non-linear Burgers' equation with the LINEAR 1D Heat Equation:

        du/dt = alpha * d²u/dx²       (alpha = 0.05)

    with exact analytical solution (used for evaluation):

        u(x, t) = sin(pi*x) * exp(-alpha * pi^2 * t)

    Boundary & Initial Conditions:
        IC : u(x, 0)  = sin(pi * x)       for x in [-1, 1]
        BC : u(-1, t) = u(1, t) = 0       for t in [0,  1]

    Train both:
        1. Classical c-PINN  (3 hidden layers × 40 neurons, Tanh)
        2. Hybrid   QAPINN   (4 qubits, 2 variational layers, lightning.qubit)

    for 3000 epochs each under IDENTICAL boundary conditions.

    Key Hypothesis:
        On a LINEAR PDE there is no shock discontinuity, so both networks
        should converge well. The question is:
        - Does QAPINN's smaller parameter count hurt or help?
        - Does quantum entanglement add value on smooth, linear domains?

Output Files:
    evaluation/heat_equation/exact_solution.npy
    evaluation/heat_equation/cpinn_pred.npy
    evaluation/heat_equation/qapinn_pred.npy
    evaluation/heat_equation/cpinn_loss_history.npy
    evaluation/heat_equation/qapinn_loss_history.npy
    evaluation/plots/heat_eq_comparison.png     <-- main output image

Run:
    python train_heat_equation.py
    python train_heat_equation.py --epochs 1000   # quick test
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.cpinn import ClassicalPINN
from src.qapinn import HybridQAPINN, count_parameters

# ── Output directories ─────────────────────────────────────────────────────────
OUT_DIR  = 'evaluation/heat_equation'
PLOT_DIR = 'evaluation/plots'
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

ALPHA     = 0.05      # thermal diffusivity
EPOCHS    = 3000      # training epochs for each model
LR        = 1e-3      # Adam learning rate
SEED      = 42

# Collocation point counts
N_F = 2000            # interior PDE points
N_I = 200             # initial condition points
N_B = 200             # boundary condition points (100 per side)

# Evaluation grid
NX = 256
NT = 100

# QAPINN settings
N_QUBITS = 4
N_LAYERS = 2
DEVICE   = 'lightning.qubit'

# Loss weights  [PDE, IC, BC]
WEIGHTS  = (1.0, 10.0, 10.0)

LOG_EVERY = 300       # print progress every N epochs

SAVE_PLOT = os.path.join(PLOT_DIR, 'heat_eq_comparison.png')

# Dark theme colours
BG      = '#0d1117'
AX_BG   = '#161b22'
GRID_C  = '#30363d'
WIRE_C  = '#c9d1d9'
TITLE_C = '#58a6ff'
PARAM_C = '#e3b341'
RED_C   = '#f85149'
GREEN_C = '#3fb950'
BLUE_C  = '#388bfd'


# =============================================================================
# HEAT EQUATION — EXACT SOLUTION & DATA SAMPLER
# =============================================================================

def exact_heat(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical solution to the 1D heat equation:
        u(x,t) = sin(pi*x) * exp(-alpha * pi^2 * t)
    """
    return np.sin(np.pi * x) * np.exp(-ALPHA * np.pi**2 * t)


def sample_data(seed: int = 42):
    """
    Sample collocation, IC, and BC points for the heat equation.
    Domain: x in [-1, 1],  t in [0, 1]
    """
    rng = np.random.default_rng(seed)

    # ── Interior (PDE collocation) ─────────────────────────────────────────
    x_f = rng.uniform(-1.0, 1.0, (N_F, 1)).astype(np.float32)
    t_f = rng.uniform( 0.0, 1.0, (N_F, 1)).astype(np.float32)

    x_f_t = torch.tensor(x_f, requires_grad=True)
    t_f_t = torch.tensor(t_f, requires_grad=True)
    X_f_t = torch.cat([x_f_t, t_f_t], dim=1)

    # ── Initial condition  u(x, 0) = sin(pi*x) ────────────────────────────
    x_i = rng.uniform(-1.0, 1.0, (N_I, 1)).astype(np.float32)
    t_i = np.zeros((N_I, 1), dtype=np.float32)
    u_i = np.sin(np.pi * x_i).astype(np.float32)

    X_i_t = torch.tensor(np.hstack([x_i, t_i]))
    u_i_t = torch.tensor(u_i)

    # ── Boundary conditions  u(-1,t) = u(1,t) = 0 ─────────────────────────
    t_bc = rng.uniform(0.0, 1.0, (N_B, 1)).astype(np.float32)
    x_left  = -np.ones((N_B // 2, 1), dtype=np.float32)
    x_right =  np.ones((N_B // 2, 1), dtype=np.float32)
    x_bc = np.vstack([x_left, x_right])
    u_bc = np.zeros((N_B, 1), dtype=np.float32)

    X_b_t = torch.tensor(np.hstack([x_bc, t_bc]))
    u_b_t = torch.tensor(u_bc)

    return x_f_t, t_f_t, X_f_t, X_i_t, u_i_t, X_b_t, u_b_t


def make_eval_grid():
    """
    Build a uniform evaluation grid and compute the exact solution.
    Returns tensors for model inference and numpy arrays for plotting.
    """
    x_vals = np.linspace(-1.0, 1.0, NX).astype(np.float32)
    t_vals = np.linspace( 0.0, 1.0, NT).astype(np.float32)
    T_grid, X_grid = np.meshgrid(t_vals, x_vals)   # both shape (NX, NT)

    x_flat = X_grid.flatten()[:, None]
    t_flat = T_grid.flatten()[:, None]
    X_eval = torch.tensor(np.hstack([x_flat, t_flat]))

    u_exact = exact_heat(X_grid, T_grid)            # shape (NX, NT)

    return X_eval, X_grid, T_grid, u_exact


# =============================================================================
# HEAT EQUATION PDE RESIDUAL
# =============================================================================

def heat_pde_residual(model, x_f: torch.Tensor,
                      t_f: torch.Tensor) -> torch.Tensor:
    """
    Computes the residual of the 1D heat equation:
        R = u_t - alpha * u_xx

    For an ideal solution R = 0 everywhere.
    """
    X_f = torch.cat([x_f, t_f], dim=1)
    u   = model(X_f)

    # First-order derivatives
    grads = torch.autograd.grad(
        outputs=u, inputs=[t_f, x_f],
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )
    u_t = grads[0]
    u_x = grads[1]

    # Second-order spatial derivative
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]

    # Heat equation residual  (linear — no u*u_x term!)
    residual = u_t - ALPHA * u_xx
    return residual


# =============================================================================
# TRAINING FUNCTION  (shared for both architectures)
# =============================================================================

def train_model(model: nn.Module,
                model_name: str,
                x_f, t_f, X_f, X_i, u_i, X_b, u_b,
                X_eval: torch.Tensor,
                u_exact: np.ndarray,
                epochs: int) -> dict:
    """
    Train `model` for `epochs` epochs on the heat equation.
    Returns dict with prediction field, loss history, and timing.
    """
    print(f"\n{'='*60}")
    print(f"  Training  {model_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs    : {epochs}   |   Alpha = {ALPHA}   |   Backend: {DEVICE}")
    print(f"{'='*60}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    w_f, w_i, w_b = WEIGHTS
    history = {'total': [], 'pde': [], 'ic': [], 'bc': []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # PDE residual loss  (heat equation)
        residual = heat_pde_residual(model, x_f, t_f)
        loss_f   = torch.mean(residual ** 2)

        # Initial condition loss
        loss_i = torch.mean((model(X_i) - u_i) ** 2)

        # Boundary condition loss
        loss_b = torch.mean((model(X_b) - u_b) ** 2)

        total = w_f * loss_f + w_i * loss_i + w_b * loss_b
        total.backward()
        optimizer.step()
        scheduler.step()

        history['total'].append(total.item())
        history['pde'].append(loss_f.item())
        history['ic'].append(loss_i.item())
        history['bc'].append(loss_b.item())

        if epoch % LOG_EVERY == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                u_pred_flat = model(X_eval).cpu().numpy().flatten()
                u_pred_2d   = u_pred_flat.reshape(u_exact.shape)
                rel_l2 = (np.linalg.norm(u_pred_2d - u_exact)
                          / np.linalg.norm(u_exact))
            print(f"  Epoch {epoch:5d}/{epochs}  "
                  f"Loss: {total.item():.4e}  "
                  f"PDE: {loss_f.item():.4e}  "
                  f"Rel L2: {rel_l2:.4e}  "
                  f"({(time.time()-t0)/60:.1f} min)")

    elapsed = time.time() - t0

    # Final prediction
    model.eval()
    with torch.no_grad():
        u_final = (model(X_eval).cpu().numpy()
                   .flatten().reshape(u_exact.shape))
    rel_l2_final = (np.linalg.norm(u_final - u_exact)
                    / np.linalg.norm(u_exact))

    print(f"\n  {model_name} DONE  |  "
          f"Rel L2: {rel_l2_final:.6f}  |  "
          f"Time: {elapsed/60:.1f} min")

    return {
        'u_pred'       : u_final,
        'history'      : history,
        'rel_l2'       : rel_l2_final,
        'n_params'     : n_params,
        'elapsed'      : elapsed,
    }


# =============================================================================
# PLOTTING HELPER
# =============================================================================

def _style_ax(ax):
    """Apply consistent dark-theme styling to an axes — compatible with all matplotlib versions."""
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=WIRE_C)
    for spine in ax.spines.values():   # compatible with matplotlib 3.3+
        spine.set_color(GRID_C)


# =============================================================================
# PLOTTING — 9-PANEL COMPARISON IMAGE
# =============================================================================

def plot_results(cpinn_res: dict, qapinn_res: dict,
                 X_grid, T_grid, u_exact: np.ndarray,
                 epochs: int):
    """
    Generates a 3-row × 3-column panel figure and saves it to disk.
    Row 0: Exact | c-PINN prediction | QAPINN prediction
    Row 1: c-PINN error | QAPINN error | PDE loss trace
    Row 2: Total loss | IC+BC loss | Parameter bar chart
    """
    import traceback

    print("  Building figure ...")
    try:
        fig = plt.figure(figsize=(21, 14))
        fig.patch.set_facecolor(BG)
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

        vmin, vmax = float(u_exact.min()), float(u_exact.max())
        # Guard: if all values same (degenerate), pad slightly
        if abs(vmax - vmin) < 1e-8:
            vmin -= 0.1
            vmax += 0.1
        norm_sol = Normalize(vmin=vmin, vmax=vmax)
        cmap_sol = 'plasma'
        cmap_err = 'hot'

        # ── Row 0: Solution fields ───────────────────────────────────────────
        row0_data = [
            ('Exact Solution\nu(x,t) = sin(px)*exp(-ap^2*t)', TITLE_C, u_exact),
            (f"c-PINN Prediction\nRel L2 = {cpinn_res['rel_l2']:.4f} | "
             f"{cpinn_res['n_params']:,} params",  BLUE_C,  cpinn_res['u_pred']),
            (f"QAPINN Prediction\nRel L2 = {qapinn_res['rel_l2']:.4f} | "
             f"{qapinn_res['n_params']:,} params", GREEN_C, qapinn_res['u_pred']),
        ]
        for col, (title, color, field) in enumerate(row0_data):
            ax = fig.add_subplot(gs[0, col])
            _style_ax(ax)
            im = ax.pcolormesh(T_grid, X_grid, field,
                               cmap=cmap_sol, norm=norm_sol, shading='auto')
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
            cb.outline.set_edgecolor(GRID_C)
            ax.set_title(title, color=color, fontsize=10, fontweight='bold', pad=8)
            ax.set_xlabel('Time  t',  color=WIRE_C, fontsize=9)
            ax.set_ylabel('Space  x', color=WIRE_C, fontsize=9)

        # ── Row 1: Error maps + PDE loss ─────────────────────────────────────
        for col, (res, label, color) in enumerate([
                (cpinn_res, 'c-PINN', BLUE_C),
                (qapinn_res,'QAPINN', GREEN_C)]):
            ax = fig.add_subplot(gs[1, col])
            _style_ax(ax)
            err    = np.abs(res['u_pred'] - u_exact)
            vmax_e = float(np.percentile(err, 98)) if err.max() > 0 else 1.0
            im_e   = ax.pcolormesh(T_grid, X_grid, err,
                                   cmap=cmap_err, shading='auto',
                                   vmin=0, vmax=vmax_e)
            cb_e   = fig.colorbar(im_e, ax=ax, pad=0.02)
            cb_e.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
            cb_e.outline.set_edgecolor(GRID_C)
            ax.set_title(f'|Error|  {label}', color=RED_C,
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Time  t',  color=WIRE_C, fontsize=9)
            ax.set_ylabel('Space  x', color=WIRE_C, fontsize=9)

        # PDE loss trace (col 2)
        ep      = list(range(1, epochs + 1))
        ax_pde  = fig.add_subplot(gs[1, 2])
        _style_ax(ax_pde)
        ax_pde.set_yscale('log')
        ax_pde.plot(ep, cpinn_res['history']['pde'],
                    color=BLUE_C,  lw=1.8, label='c-PINN PDE loss')
        ax_pde.plot(ep, qapinn_res['history']['pde'],
                    color=GREEN_C, lw=1.8, label='QAPINN PDE loss')
        ax_pde.set_title('PDE Residual Loss  du/dt - a*d2u/dx2',
                         color=TITLE_C, fontsize=10, fontweight='bold')
        ax_pde.set_xlabel('Epoch',          color=WIRE_C, fontsize=9)
        ax_pde.set_ylabel('PDE Loss [log]', color=WIRE_C, fontsize=9)
        ax_pde.grid(True, color=GRID_C, lw=0.6, alpha=0.7, which='both')
        ax_pde.legend(fontsize=9, facecolor=BG, edgecolor=GRID_C, labelcolor=WIRE_C)

        # ── Row 2: Total loss | IC+BC loss | Bar chart ───────────────────────
        # Total loss
        ax_tot = fig.add_subplot(gs[2, 0])
        _style_ax(ax_tot)
        ax_tot.set_yscale('log')
        ax_tot.plot(ep, cpinn_res['history']['total'],
                    color=BLUE_C,  lw=1.8,
                    label=f"c-PINN  final={cpinn_res['history']['total'][-1]:.3e}")
        ax_tot.plot(ep, qapinn_res['history']['total'],
                    color=GREEN_C, lw=1.8,
                    label=f"QAPINN  final={qapinn_res['history']['total'][-1]:.3e}")
        ax_tot.set_title('Total Training Loss [log scale]',
                         color=TITLE_C, fontsize=10, fontweight='bold')
        ax_tot.set_xlabel('Epoch',       color=WIRE_C, fontsize=9)
        ax_tot.set_ylabel('Total Loss',  color=WIRE_C, fontsize=9)
        ax_tot.grid(True, color=GRID_C, lw=0.6, alpha=0.7, which='both')
        ax_tot.legend(fontsize=8, facecolor=BG, edgecolor=GRID_C, labelcolor=WIRE_C)

        # IC + BC loss
        ax_bc = fig.add_subplot(gs[2, 1])
        _style_ax(ax_bc)
        ax_bc.set_yscale('log')
        ax_bc.plot(ep, cpinn_res['history']['ic'],
                   color=BLUE_C,  lw=1.5, ls='-',  label='c-PINN IC')
        ax_bc.plot(ep, cpinn_res['history']['bc'],
                   color=BLUE_C,  lw=1.5, ls='--', label='c-PINN BC')
        ax_bc.plot(ep, qapinn_res['history']['ic'],
                   color=GREEN_C, lw=1.5, ls='-',  label='QAPINN IC')
        ax_bc.plot(ep, qapinn_res['history']['bc'],
                   color=GREEN_C, lw=1.5, ls='--', label='QAPINN BC')
        ax_bc.set_title('IC & BC Loss Convergence',
                        color=TITLE_C, fontsize=10, fontweight='bold')
        ax_bc.set_xlabel('Epoch',       color=WIRE_C, fontsize=9)
        ax_bc.set_ylabel('Loss [log]',  color=WIRE_C, fontsize=9)
        ax_bc.grid(True, color=GRID_C, lw=0.6, alpha=0.7, which='both')
        ax_bc.legend(fontsize=7.5, facecolor=BG, edgecolor=GRID_C,
                     labelcolor=WIRE_C, ncol=2)

        # Parameter bar chart
        ax_bar  = fig.add_subplot(gs[2, 2])
        _style_ax(ax_bar)
        bar_labels  = ['c-PINN', 'QAPINN']
        bar_params  = [cpinn_res['n_params'], qapinn_res['n_params']]
        bar_rel_l2s = [cpinn_res['rel_l2'],   qapinn_res['rel_l2']]
        bar_colors  = [BLUE_C, GREEN_C]
        bars        = ax_bar.bar(bar_labels, bar_params,
                                 color=bar_colors, width=0.4,
                                 edgecolor='white', linewidth=1.2)
        for bar, p, rl2, col in zip(bars, bar_params, bar_rel_l2s, bar_colors):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 20,
                        f'{p:,} params\nRel L2={rl2:.4f}',
                        ha='center', va='bottom',
                        fontsize=9, color=col, fontweight='bold')
        ax_bar.set_title('Parameter Count vs Accuracy\n(Heat Equation - Linear Domain)',
                         color=TITLE_C, fontsize=10, fontweight='bold')
        ax_bar.set_ylabel('Trainable Parameters', color=WIRE_C, fontsize=9)
        ax_bar.grid(True, color=GRID_C, lw=0.6, alpha=0.5, axis='y')
        ax_bar.set_ylim(0, max(bar_params) * 1.35)

        # ── Main title & footer ──────────────────────────────────────────────
        fig.suptitle(
            f'c-PINN vs QAPINN  -  1D Linear Heat Equation  (alpha = {ALPHA})\n'
            f'{epochs} Epochs  |  {N_QUBITS} Qubits  |  Backend: {DEVICE}  |  '
            f'WISER Summer Program 2026',
            color=TITLE_C, fontsize=14, fontweight='bold', y=0.99)

        fig.text(0.5, 0.003,
                 f'PDE: du/dt = {ALPHA} * d2u/dx2   |   '
                 f'IC: u(x,0) = sin(pi*x)   |   '
                 f'BC: u(-1,t) = u(1,t) = 0   |   '
                 f'Exact: u(x,t) = sin(pi*x)*exp(-{ALPHA}*pi^2*t)',
                 ha='center', fontsize=8.5, color=PARAM_C)

        # ── Save ─────────────────────────────────────────────────────────────
        print(f"  Saving figure to {SAVE_PLOT} ...")
        plt.savefig(SAVE_PLOT, dpi=160, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        # Confirm file was actually written
        if os.path.isfile(SAVE_PLOT):
            size_kb = os.path.getsize(SAVE_PLOT) // 1024
            print(f"  Plot saved successfully  ({size_kb} KB)  ->  {SAVE_PLOT}")
        else:
            print(f"  [!] WARNING: File was NOT created at {SAVE_PLOT}")
            return

        # Auto-open in Windows photo viewer
        try:
            subprocess.Popen(['start', os.path.abspath(SAVE_PLOT)], shell=True)
            print("  Image opened in your default photo viewer!")
        except Exception:
            print(f"  Open manually: {os.path.abspath(SAVE_PLOT)}")

    except Exception:
        print("\n  [!] PLOTTING FAILED - full error below:")
        traceback.print_exc()
        print("  Training results are still saved in evaluation/heat_equation/")



# =============================================================================
# MAIN
# =============================================================================

def main(epochs: int):
    print()
    print("+============================================================+")
    print("|  1D LINEAR HEAT EQUATION - c-PINN vs QAPINN               |")
    print(f"|  du/dt = {ALPHA} * d2u/dx2                                  |")
    print(f"|  IC: u(x,0) = sin(pi*x)  |  BC: u(+-1,t) = 0               |")
    print(f"|  Epochs: {epochs}   |   Alpha: {ALPHA}   |   Seed: {SEED}              |")
    print("+============================================================+")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Build data ─────────────────────────────────────────────────────────
    print("\n  Sampling training data ...")
    x_f, t_f, X_f, X_i, u_i, X_b, u_b = sample_data(seed=SEED)

    print("  Building evaluation grid ...")
    X_eval, X_grid, T_grid, u_exact = make_eval_grid()

    # ── Save exact solution ────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, 'exact_solution.npy'), u_exact)
    np.save(os.path.join(OUT_DIR, 'x_grid.npy'), X_grid)
    np.save(os.path.join(OUT_DIR, 't_grid.npy'), T_grid)
    print(f"  Exact solution shape: {u_exact.shape}  "
          f"[range {u_exact.min():.3f} to {u_exact.max():.3f}]")

    # ── Train c-PINN ──────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    cpinn = ClassicalPINN(in_features=2, hidden_features=40,
                          out_features=1, num_layers=3)
    cpinn_res = train_model(
        cpinn, 'Classical c-PINN',
        x_f, t_f, X_f, X_i, u_i, X_b, u_b,
        X_eval, u_exact, epochs)

    np.save(os.path.join(OUT_DIR, 'cpinn_pred.npy'),         cpinn_res['u_pred'])
    np.save(os.path.join(OUT_DIR, 'cpinn_loss_history.npy'), cpinn_res['history']['total'])
    print(f"  c-PINN results saved to {OUT_DIR}/")

    # ── Train QAPINN ──────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    qapinn = HybridQAPINN(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        position='first',
        measurement_mode='expectation',
        encoding='angle',
        entanglement='linear',
        hidden_features=40,
        device_name=DEVICE
    )
    qapinn_res = train_model(
        qapinn, f'Hybrid QAPINN ({N_QUBITS} qubits, {DEVICE})',
        x_f, t_f, X_f, X_i, u_i, X_b, u_b,
        X_eval, u_exact, epochs)

    np.save(os.path.join(OUT_DIR, 'qapinn_pred.npy'),         qapinn_res['u_pred'])
    np.save(os.path.join(OUT_DIR, 'qapinn_loss_history.npy'), qapinn_res['history']['total'])
    print(f"  QAPINN results saved to {OUT_DIR}/")

    # ── Console summary ────────────────────────────────────────────────────
    print()
    print("="*65)
    print(f"  FINAL RESULTS — Heat Equation  (alpha={ALPHA}, {epochs} epochs)")
    print("="*65)
    print(f"  {'Model':>15}  {'Params':>8}  "
          f"{'Rel L2 Error':>14}  {'Time (min)':>10}")
    print("  " + "-"*55)
    for name, res in [('c-PINN', cpinn_res), ('QAPINN', qapinn_res)]:
        print(f"  {name:>15}  {res['n_params']:>8,}  "
              f"{res['rel_l2']:>14.6f}  {res['elapsed']/60:>10.1f}")
    print("="*65)

    better = ('c-PINN' if cpinn_res['rel_l2'] < qapinn_res['rel_l2']
              else 'QAPINN')
    diff   = abs(cpinn_res['rel_l2'] - qapinn_res['rel_l2'])
    print(f"\n  Winner on linear domain: {better}  "
          f"(margin: {diff:.4f} Rel L2)")
    print(f"\n  Key Insight:")
    if cpinn_res['rel_l2'] < qapinn_res['rel_l2']:
        print("  Classical PINN wins on the linear heat equation.")
        print("  Quantum entanglement adds no advantage for smooth linear fields.")
        print("  This confirms: quantum advantage is specific to complex, non-linear PDEs.")
    else:
        print("  QAPINN matches or beats c-PINN even on linear domains.")
        print("  Quantum feature maps may provide useful inductive bias even here.")

    # ── Generate comparison plot ───────────────────────────────────────────
    print("\n  Generating comparison plot ...")
    plot_results(cpinn_res, qapinn_res, X_grid, T_grid, u_exact, epochs)

    print("\n  DONE.")
    print(f"  All data saved to : {OUT_DIR}/")
    print(f"  Plot saved to     : {SAVE_PLOT}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Heat Equation: c-PINN vs QAPINN — WISER 2026')
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Training epochs (default: {EPOCHS})')
    args = parser.parse_args()
    main(args.epochs)
