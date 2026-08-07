"""
train_wave_equation.py
======================
WISER Summer Program 2026 — Hyperbolic PDE Experiment

Experiment:
    Solve the 1D Wave Equation using QAPINN:

        d²u/dt² = c² * d²u/dx²       (c = 1.0)

    with exact analytical solution:

        u(x, t) = sin(pi*x) * cos(pi*c*t)

    Boundary & Initial Conditions:
        IC1: u(x, 0)   = sin(pi*x)    displacement   (sinusoidal profile)
        IC2: du/dt(x,0) = 0            zero initial velocity
        BC : u(-1, t)  = u(1, t) = 0  Dirichlet walls

    Key Challenge vs Heat Equation:
        The wave equation is HYPERBOLIC — it has a SECOND-ORDER time derivative.
        The PDE residual requires computing u_tt via two nested autograd passes:
            1st pass:  u   -> u_t  (grad w.r.t. t, create_graph=True)
            2nd pass:  u_t -> u_tt (grad w.r.t. t again, create_graph=True)

    Architecture:
        Hybrid QAPINN — 4 qubits, 2 variational layers, lightning.qubit (CPU C++)

Output Files:
    evaluation/wave_equation/exact_solution.npy
    evaluation/wave_equation/qapinn_pred.npy
    evaluation/wave_equation/qapinn_loss_history.npy
    evaluation/plots/wave_eq_qapinn.png     <-- main output image (auto-opens)

Run:
    python train_wave_equation.py
    python train_wave_equation.py --epochs 1000   # quick test
"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.qapinn import HybridQAPINN, count_parameters

# ── Output directories ─────────────────────────────────────────────────────────
OUT_DIR  = 'evaluation/wave_equation'
PLOT_DIR = 'evaluation/plots'
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SAVE_PLOT = os.path.join(PLOT_DIR, 'wave_eq_qapinn.png')


# =============================================================================
# CONFIGURATION
# =============================================================================

C         = 1.0       # wave speed
EPOCHS    = 5000      # training epochs
LR        = 1e-3      # Adam learning rate
SEED      = 42

# Collocation point counts
N_F = 3000            # interior PDE points  (more needed for 2nd-order PDE)
N_I = 300             # initial condition points (u and u_t separately)
N_B = 200             # boundary condition points

# Evaluation grid
NX = 256
NT = 150

# QAPINN settings
N_QUBITS = 4
N_LAYERS = 2
DEVICE   = 'lightning.qubit'

# Loss weights  [PDE, IC displacement, IC velocity, BC]
# IC velocity weighted higher — harder to enforce via autograd
W_PDE = 1.0
W_IC1 = 10.0    # displacement u(x,0)
W_IC2 = 5.0     # velocity    du/dt(x,0) = 0
W_BC  = 10.0

LOG_EVERY = 500   # print progress every N epochs

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
PURP_C  = '#bc8cff'
ORNG_C  = '#d29922'


# =============================================================================
# EXACT SOLUTION & DATA SAMPLING
# =============================================================================

def exact_wave(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical solution to the 1D wave equation:
        u(x,t) = sin(pi*x) * cos(pi*c*t)
    Valid for IC: u(x,0) = sin(pi*x),  du/dt(x,0) = 0,  BC: u(+-1,t) = 0
    """
    return np.sin(np.pi * x) * np.cos(np.pi * C * t)


def sample_data(seed: int = 42):
    """
    Sample collocation, IC (displacement + velocity), and BC points.
    Domain: x in [-1, 1],  t in [0, 1]
    Returns all tensors with requires_grad where needed.
    """
    rng = np.random.default_rng(seed)

    # ── Interior PDE collocation points ───────────────────────────────────────
    x_f_np = rng.uniform(-1.0, 1.0, (N_F, 1)).astype(np.float32)
    t_f_np = rng.uniform( 0.0, 1.0, (N_F, 1)).astype(np.float32)

    x_f = torch.tensor(x_f_np, requires_grad=True)
    t_f = torch.tensor(t_f_np, requires_grad=True)

    # ── IC1: Displacement  u(x,0) = sin(pi*x) ────────────────────────────────
    x_i_np = rng.uniform(-1.0, 1.0, (N_I, 1)).astype(np.float32)
    t_i_np = np.zeros((N_I, 1), dtype=np.float32)
    u_i_np = np.sin(np.pi * x_i_np).astype(np.float32)

    X_ic1 = torch.tensor(np.hstack([x_i_np, t_i_np]))
    u_ic1 = torch.tensor(u_i_np)

    # ── IC2: Velocity  du/dt(x,0) = 0 ────────────────────────────────────────
    # Same spatial points at t=0 but need grad through t for u_t
    x_v_np = rng.uniform(-1.0, 1.0, (N_I, 1)).astype(np.float32)
    t_v_np = np.zeros((N_I, 1), dtype=np.float32)

    x_ic2 = torch.tensor(x_v_np, requires_grad=True)
    t_ic2 = torch.tensor(t_v_np, requires_grad=True)

    # ── Boundary conditions  u(-1,t) = u(1,t) = 0 ────────────────────────────
    t_bc_np = rng.uniform(0.0, 1.0, (N_B, 1)).astype(np.float32)
    x_left  = -np.ones((N_B // 2, 1), dtype=np.float32)
    x_right =  np.ones((N_B // 2, 1), dtype=np.float32)
    x_bc_np = np.vstack([x_left, x_right])
    u_bc_np = np.zeros((N_B, 1), dtype=np.float32)

    X_bc = torch.tensor(np.hstack([x_bc_np, t_bc_np]))
    u_bc = torch.tensor(u_bc_np)

    return x_f, t_f, x_ic2, t_ic2, X_ic1, u_ic1, X_bc, u_bc


def make_eval_grid():
    """Build uniform evaluation grid and compute exact solution."""
    x_vals = np.linspace(-1.0, 1.0, NX).astype(np.float32)
    t_vals = np.linspace( 0.0, 1.0, NT).astype(np.float32)
    T_grid, X_grid = np.meshgrid(t_vals, x_vals)   # (NX, NT)

    x_flat = X_grid.flatten()[:, None]
    t_flat = T_grid.flatten()[:, None]
    X_eval = torch.tensor(np.hstack([x_flat, t_flat]))

    u_exact = exact_wave(X_grid, T_grid)             # (NX, NT)

    return X_eval, X_grid, T_grid, u_exact


# =============================================================================
# WAVE EQUATION PDE RESIDUAL
# Two nested autograd passes to get u_tt (second-order time derivative)
# =============================================================================

def wave_pde_residual(model, x_f: torch.Tensor,
                      t_f: torch.Tensor) -> torch.Tensor:
    """
    Computes the 1D wave equation residual:
        R = u_tt - c^2 * u_xx

    Requires TWO autograd passes through time to get u_tt:
        Pass 1: u    -> u_t    (create_graph=True to allow Pass 2)
        Pass 2: u_t  -> u_tt   (create_graph=True for loss.backward())

    And TWO passes through space to get u_xx:
        Pass 1: u    -> u_x
        Pass 2: u_x  -> u_xx
    """
    X_f = torch.cat([x_f, t_f], dim=1)
    u   = model(X_f)
    ones = torch.ones_like(u)

    # ── First-order derivatives ───────────────────────────────────────────────
    grads = torch.autograd.grad(
        outputs=u, inputs=[x_f, t_f],
        grad_outputs=ones,
        create_graph=True, retain_graph=True
    )
    u_x = grads[0]
    u_t = grads[1]

    # ── Second-order derivatives ──────────────────────────────────────────────
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]

    u_tt = torch.autograd.grad(
        outputs=u_t, inputs=t_f,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True, retain_graph=True
    )[0]

    # Wave residual: u_tt - c^2 * u_xx = 0
    residual = u_tt - (C ** 2) * u_xx
    return residual


def velocity_residual(model, x_ic2: torch.Tensor,
                      t_ic2: torch.Tensor) -> torch.Tensor:
    """
    Computes du/dt at t=0 (IC2: zero initial velocity).
    Requires one autograd pass to get u_t at initial time slice.
    """
    X_ic2 = torch.cat([x_ic2, t_ic2], dim=1)
    u     = model(X_ic2)
    u_t   = torch.autograd.grad(
        outputs=u, inputs=t_ic2,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    return u_t


# =============================================================================
# AXIS STYLING HELPER  (compatible with all matplotlib versions)
# =============================================================================

def _style_ax(ax, grid=False, which='both'):
    """Dark-theme axis style — uses .values() not [:] for spine iteration."""
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=WIRE_C)
    for spine in ax.spines.values():
        spine.set_color(GRID_C)
    if grid:
        ax.grid(True, color=GRID_C, lw=0.6, alpha=0.7, which=which)


# =============================================================================
# TRAINING
# =============================================================================

def train_qapinn_wave(model, epochs: int) -> dict:
    """
    Train QAPINN on the 1D wave equation for `epochs` epochs.
    Returns dict with prediction, history, timing, and parameter count.
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_params  = sum(p.numel()
                    for name, p in model.named_parameters()
                    if 'quantum' in name.lower() or 'qlayer' in name.lower())

    print(f"\n{'='*65}")
    print(f"  QAPINN Training  -  1D Wave Equation")
    print(f"  c (wave speed)  : {C}")
    print(f"  PDE type        : Hyperbolic (second-order in time)")
    print(f"  Params total    : {n_params:,}  |  Quantum: {q_params}")
    print(f"  Epochs          : {epochs}")
    print(f"  Backend         : {DEVICE}")
    print(f"  Loss weights    : PDE={W_PDE}  IC_disp={W_IC1}  "
          f"IC_vel={W_IC2}  BC={W_BC}")
    print(f"{'='*65}")

    # Sample data
    x_f, t_f, x_ic2, t_ic2, X_ic1, u_ic1, X_bc, u_bc = sample_data(SEED)

    # Eval grid
    X_eval, X_grid, T_grid, u_exact = make_eval_grid()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=5e-6)

    history = {'total': [], 'pde': [], 'ic1': [], 'ic2': [], 'bc': []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1. PDE residual  (u_tt - c^2 * u_xx = 0)
        res_pde = wave_pde_residual(model, x_f, t_f)
        loss_pde = torch.mean(res_pde ** 2)

        # 2. IC1: Displacement  u(x,0) = sin(pi*x)
        loss_ic1 = torch.mean((model(X_ic1) - u_ic1) ** 2)

        # 3. IC2: Zero initial velocity  du/dt(x,0) = 0
        res_vel  = velocity_residual(model, x_ic2, t_ic2)
        loss_ic2 = torch.mean(res_vel ** 2)

        # 4. BC: u(+-1, t) = 0
        loss_bc = torch.mean((model(X_bc) - u_bc) ** 2)

        # Composite loss
        total = (W_PDE * loss_pde
                 + W_IC1 * loss_ic1
                 + W_IC2 * loss_ic2
                 + W_BC  * loss_bc)

        total.backward()
        optimizer.step()
        scheduler.step()

        history['total'].append(total.item())
        history['pde'].append(loss_pde.item())
        history['ic1'].append(loss_ic1.item())
        history['ic2'].append(loss_ic2.item())
        history['bc'].append(loss_bc.item())

        if epoch % LOG_EVERY == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                u_pred_2d = (model(X_eval).cpu().numpy()
                             .flatten().reshape(u_exact.shape))
                rel_l2    = (np.linalg.norm(u_pred_2d - u_exact)
                             / np.linalg.norm(u_exact))
            elapsed = (time.time() - t0) / 60
            print(f"  Epoch {epoch:5d}/{epochs}  "
                  f"Loss: {total.item():.4e}  "
                  f"PDE: {loss_pde.item():.4e}  "
                  f"IC_vel: {loss_ic2.item():.4e}  "
                  f"Rel L2: {rel_l2:.4e}  "
                  f"({elapsed:.1f} min)")

    elapsed_total = time.time() - t0

    # Final prediction
    model.eval()
    with torch.no_grad():
        u_final = (model(X_eval).cpu().numpy()
                   .flatten().reshape(u_exact.shape))
    rel_l2_final = (np.linalg.norm(u_final - u_exact)
                    / np.linalg.norm(u_exact))

    print(f"\n  Training DONE  |  "
          f"Rel L2: {rel_l2_final:.6f}  |  "
          f"Time: {elapsed_total/60:.1f} min")

    return {
        'u_pred'   : u_final,
        'u_exact'  : u_exact,
        'X_grid'   : X_grid,
        'T_grid'   : T_grid,
        'history'  : history,
        'rel_l2'   : rel_l2_final,
        'n_params' : n_params,
        'q_params' : q_params,
        'elapsed'  : elapsed_total,
        'epochs'   : epochs,
    }


# =============================================================================
# PLOTTING — 6-PANEL OUTPUT IMAGE
# =============================================================================

def plot_results(res: dict):
    """
    Generates a 2-row × 3-column figure:
      Row 0: Exact solution | QAPINN prediction | Absolute error contour (KEY)
      Row 1: Total loss log | PDE + IC + BC losses | Time-slice cross-sections
    """
    print("  Building figure ...")
    try:
        u_exact  = res['u_exact']
        u_pred   = res['u_pred']
        X_grid   = res['X_grid']
        T_grid   = res['T_grid']
        history  = res['history']
        epochs   = res['epochs']
        rel_l2   = res['rel_l2']
        n_params = res['n_params']
        q_params = res['q_params']

        abs_err  = np.abs(u_pred - u_exact)

        fig = plt.figure(figsize=(21, 12))
        fig.patch.set_facecolor(BG)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

        # Shared colour scale for solution panels
        vabs     = max(abs(float(u_exact.min())), abs(float(u_exact.max())))
        norm_sol = Normalize(vmin=-vabs, vmax=vabs)
        cmap_sol = 'RdBu_r'    # red-blue diverging — perfect for wave
        cmap_err = 'inferno'

        ep = list(range(1, epochs + 1))

        # ── [0,0] Exact solution ──────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        _style_ax(ax0)
        im0 = ax0.pcolormesh(T_grid, X_grid, u_exact,
                             cmap=cmap_sol, norm=norm_sol, shading='auto')
        cb0 = fig.colorbar(im0, ax=ax0, pad=0.02)
        cb0.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
        cb0.outline.set_edgecolor(GRID_C)
        ax0.set_title(f'Exact Solution\nu(x,t) = sin(pi*x)*cos(pi*c*t)  [c={C}]',
                      color=TITLE_C, fontsize=10, fontweight='bold', pad=8)
        ax0.set_xlabel('Time  t',  color=WIRE_C, fontsize=9)
        ax0.set_ylabel('Space  x', color=WIRE_C, fontsize=9)
        ax0.contour(T_grid, X_grid, u_exact, levels=8,
                    colors=WIRE_C, linewidths=0.4, alpha=0.3)

        # ── [0,1] QAPINN prediction ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        _style_ax(ax1)
        im1 = ax1.pcolormesh(T_grid, X_grid, u_pred,
                             cmap=cmap_sol, norm=norm_sol, shading='auto')
        cb1 = fig.colorbar(im1, ax=ax1, pad=0.02)
        cb1.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
        cb1.outline.set_edgecolor(GRID_C)
        ax1.set_title(f'QAPINN Prediction\nRel L2 = {rel_l2:.4f}  |  '
                      f'{n_params:,} params  ({q_params} quantum)',
                      color=GREEN_C, fontsize=10, fontweight='bold', pad=8)
        ax1.set_xlabel('Time  t',  color=WIRE_C, fontsize=9)
        ax1.set_ylabel('Space  x', color=WIRE_C, fontsize=9)
        ax1.contour(T_grid, X_grid, u_pred, levels=8,
                    colors=WIRE_C, linewidths=0.4, alpha=0.3)

        # ── [0,2] Absolute error contour  (THE KEY OUTPUT) ───────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        _style_ax(ax2)
        vmax_err = float(np.percentile(abs_err, 99)) if abs_err.max() > 0 else 1.0
        im2 = ax2.pcolormesh(T_grid, X_grid, abs_err,
                             cmap=cmap_err, shading='auto',
                             vmin=0, vmax=vmax_err)
        # Overlay contour lines to highlight error topology
        cs2 = ax2.contour(T_grid, X_grid, abs_err,
                          levels=6, colors='white', linewidths=0.5, alpha=0.5)
        ax2.clabel(cs2, fmt='%.3f', colors='white', fontsize=6)
        cb2 = fig.colorbar(im2, ax=ax2, pad=0.02)
        cb2.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
        cb2.outline.set_edgecolor(GRID_C)
        ax2.set_title(f'Absolute Error  |u_QAPINN - u_exact|\n'
                      f'Max Error = {abs_err.max():.4f}  |  '
                      f'Mean Error = {abs_err.mean():.4f}',
                      color=RED_C, fontsize=10, fontweight='bold', pad=8)
        ax2.set_xlabel('Time  t',  color=WIRE_C, fontsize=9)
        ax2.set_ylabel('Space  x', color=WIRE_C, fontsize=9)

        # ── [1,0] Total loss history ──────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        _style_ax(ax3, grid=True)
        ax3.set_yscale('log')
        ax3.plot(ep, history['total'], color=GREEN_C, lw=2.0,
                 label=f"Total  final={history['total'][-1]:.3e}")
        ax3.set_title('Total Loss Convergence  [log]',
                      color=TITLE_C, fontsize=10, fontweight='bold')
        ax3.set_xlabel('Epoch',      color=WIRE_C, fontsize=9)
        ax3.set_ylabel('Total Loss', color=WIRE_C, fontsize=9)
        ax3.legend(fontsize=9, facecolor=BG, edgecolor=GRID_C, labelcolor=WIRE_C)

        # ── [1,1] PDE / IC / BC loss breakdown ───────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        _style_ax(ax4, grid=True)
        ax4.set_yscale('log')
        ax4.plot(ep, history['pde'], color=BLUE_C,  lw=1.6, ls='-',
                 label=f"PDE residual    final={history['pde'][-1]:.3e}")
        ax4.plot(ep, history['ic1'], color=GREEN_C, lw=1.6, ls='--',
                 label=f"IC displacement final={history['ic1'][-1]:.3e}")
        ax4.plot(ep, history['ic2'], color=ORNG_C,  lw=1.6, ls=':',
                 label=f"IC velocity     final={history['ic2'][-1]:.3e}")
        ax4.plot(ep, history['bc'],  color=PURP_C,  lw=1.6, ls='-.',
                 label=f"BC walls        final={history['bc'][-1]:.3e}")
        ax4.set_title('Loss Components Breakdown  [log]',
                      color=TITLE_C, fontsize=10, fontweight='bold')
        ax4.set_xlabel('Epoch',     color=WIRE_C, fontsize=9)
        ax4.set_ylabel('Loss [log]',color=WIRE_C, fontsize=9)
        ax4.legend(fontsize=7.5, facecolor=BG, edgecolor=GRID_C,
                   labelcolor=WIRE_C, loc='upper right')

        # ── [1,2] Time-slice cross-sections ──────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 2])
        _style_ax(ax5, grid=True)

        x_vals_1d = X_grid[:, 0]    # x array (NX,)
        t_vals_1d = T_grid[0, :]    # t array (NT,)
        slices    = [0.0, 0.25, 0.5, 0.75, 1.0]
        slice_colors = [BLUE_C, GREEN_C, ORNG_C, PURP_C, RED_C]

        for t_slice, sc in zip(slices, slice_colors):
            # Find nearest column in the grid
            t_idx   = int(np.argmin(np.abs(t_vals_1d - t_slice)))
            t_actual = float(t_vals_1d[t_idx])

            u_ex_sl = u_exact[:, t_idx]
            u_pr_sl = u_pred[:, t_idx]

            ax5.plot(x_vals_1d, u_ex_sl, color=sc, lw=1.8, ls='-',
                     label=f't={t_actual:.2f} exact')
            ax5.plot(x_vals_1d, u_pr_sl, color=sc, lw=1.2, ls='--',
                     alpha=0.85)

        ax5.set_title('Time-Slice Cross-Sections\n'
                      'Solid=Exact  Dashed=QAPINN',
                      color=TITLE_C, fontsize=10, fontweight='bold')
        ax5.set_xlabel('Space  x', color=WIRE_C, fontsize=9)
        ax5.set_ylabel('u(x, t)',  color=WIRE_C, fontsize=9)
        ax5.legend(fontsize=7, facecolor=BG, edgecolor=GRID_C,
                   labelcolor=WIRE_C, ncol=2)

        # ── Main title & footer ───────────────────────────────────────────────
        fig.suptitle(
            f'QAPINN  -  1D Wave Equation  d2u/dt2 = c^2 * d2u/dx2  (c = {C})\n'
            f'{epochs} Epochs  |  {N_QUBITS} Qubits  |  Backend: {DEVICE}  |  '
            f'Rel L2 Error = {rel_l2:.4f}  |  WISER Summer Program 2026',
            color=TITLE_C, fontsize=13, fontweight='bold', y=0.995)

        fig.text(0.5, 0.002,
                 f'IC: u(x,0)=sin(pi*x),  du/dt(x,0)=0   |   '
                 f'BC: u(-1,t)=u(1,t)=0   |   '
                 f'Exact: u(x,t)=sin(pi*x)*cos(pi*c*t)   |   '
                 f'PDE type: Hyperbolic   |   '
                 f'Params: {n_params:,} total  ({q_params} quantum)',
                 ha='center', fontsize=8, color=PARAM_C)

        # ── Save ─────────────────────────────────────────────────────────────
        print(f"  Saving figure to  {SAVE_PLOT} ...")
        plt.savefig(SAVE_PLOT, dpi=160, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        if os.path.isfile(SAVE_PLOT):
            size_kb = os.path.getsize(SAVE_PLOT) // 1024
            print(f"  Plot saved  ({size_kb} KB)  ->  {SAVE_PLOT}")
        else:
            print(f"  [!] WARNING: file was NOT written at {SAVE_PLOT}")
            return

        try:
            subprocess.Popen(['start', os.path.abspath(SAVE_PLOT)], shell=True)
            print("  Image opened in your default photo viewer!")
        except Exception:
            print(f"  Open manually: {os.path.abspath(SAVE_PLOT)}")

    except Exception:
        print("\n  [!] PLOTTING FAILED - full error below:")
        traceback.print_exc()
        print("  Training data is still saved in evaluation/wave_equation/")


# =============================================================================
# MAIN
# =============================================================================

def main(epochs: int):
    print()
    print("+================================================================+")
    print("|  1D WAVE EQUATION  -  QAPINN                                  |")
    print(f"|  d2u/dt2 = {C}^2 * d2u/dx2                                    |")
    print(f"|  IC: u(x,0)=sin(pi*x)  |  du/dt(x,0)=0  |  BC: u(+-1,t)=0  |")
    print(f"|  Epochs : {epochs}   |   c = {C}   |   Qubits = {N_QUBITS}                |")
    print("+================================================================+")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Build QAPINN ──────────────────────────────────────────────────────────
    model = HybridQAPINN(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        position='first',
        measurement_mode='expectation',
        encoding='angle',
        entanglement='linear',
        hidden_features=40,
        device_name=DEVICE
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    res = train_qapinn_wave(model, epochs)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, 'exact_solution.npy'),    res['u_exact'])
    np.save(os.path.join(OUT_DIR, 'qapinn_pred.npy'),       res['u_pred'])
    np.save(os.path.join(OUT_DIR, 'qapinn_loss_total.npy'), res['history']['total'])
    np.save(os.path.join(OUT_DIR, 'qapinn_loss_pde.npy'),   res['history']['pde'])
    np.save(os.path.join(OUT_DIR, 'qapinn_loss_ic2.npy'),   res['history']['ic2'])
    print(f"\n  Arrays saved to  {OUT_DIR}/")

    # ── Console summary ───────────────────────────────────────────────────────
    abs_err = np.abs(res['u_pred'] - res['u_exact'])
    print()
    print("=" * 65)
    print(f"  FINAL RESULTS — Wave Equation  (c={C}, {epochs} epochs)")
    print("=" * 65)
    print(f"  Rel L2 Error    : {res['rel_l2']:.6f}")
    print(f"  Max |error|     : {abs_err.max():.6f}")
    print(f"  Mean |error|    : {abs_err.mean():.6f}")
    print(f"  Total params    : {res['n_params']:,}")
    print(f"  Quantum params  : {res['q_params']}")
    print(f"  Training time   : {res['elapsed']/60:.1f} min")
    print("=" * 65)
    print()
    print("  PDE Type: HYPERBOLIC (wave equation)")
    print("  Key Difference vs Heat Equation:")
    print("    Heat  -> one time derivative  (u_t)    : parabolic, dissipative")
    print("    Wave  -> TWO time derivatives (u_tt)   : hyperbolic, oscillatory")
    print("    Both needing zero at BC walls")
    print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("  Generating output plot ...")
    plot_results(res)

    print()
    print(f"  DONE. Plot -> {SAVE_PLOT}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='1D Wave Equation — QAPINN — WISER 2026')
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Training epochs (default: {EPOCHS})')
    args = parser.parse_args()
    main(args.epochs)
