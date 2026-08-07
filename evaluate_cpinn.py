# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import os
# pyrefly: ignore [missing-import]
import matplotlib.gridspec as gridspec

def evaluate_and_plot():
    print("==================================================================")
    print("Generating Part 1 XAI Visualizations and Evaluation Metrics...")
    print("==================================================================")

    # 1. Load Data
    u_exact = np.load('evaluation/exact_solution.npy')
    u_pred = np.load('evaluation/cpinn_predictions.npy')
    x_grid = np.load('evaluation/x_grid.npy')
    t_grid = np.load('evaluation/t_grid.npy')
    history = np.load('evaluation/loss_history.npy', allow_pickle=True).item()

    X, T = np.meshgrid(x_grid, t_grid)
    abs_error = np.abs(u_exact - u_pred)

    # Calculate L2 error
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    mse_error = np.mean((u_pred - u_exact)**2)

    print(f"Final Validation Baseline Metrics:")
    print(f"  Relative L2 Error Floor: {l2_error:.4e}")
    print(f"  Mean Squared Error (MSE): {mse_error:.4e}")
    print(f"  Max Absolute Error: {abs_error.max():.4e}")

    os.makedirs('evaluation/plots', exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 1: Exact vs c-PINN Prediction & 1D Profiles
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])

    # Subplot A: Exact Solution
    ax0 = fig.add_subplot(gs[0, 0])
    c0 = ax0.pcolormesh(X, T, u_exact, cmap='plasma', shading='auto')
    ax0.set_title("Exact Solution (Cole-Hopf Analytical)", fontsize=14, fontweight='bold')
    ax0.set_xlabel("x", fontsize=12)
    ax0.set_ylabel("t", fontsize=12)
    fig.colorbar(c0, ax=ax0)

    # Subplot B: c-PINN Prediction
    ax1 = fig.add_subplot(gs[0, 1])
    c1 = ax1.pcolormesh(X, T, u_pred, cmap='plasma', shading='auto')
    ax1.set_title(f"c-PINN Prediction (Rel L2 = {l2_error:.4e})", fontsize=14, fontweight='bold')
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("t", fontsize=12)
    fig.colorbar(c1, ax=ax1)

    # Subplot C: 1D Slices over time
    ax2 = fig.add_subplot(gs[1, :])
    t_indices = [int(len(t_grid) * idx) for idx in [0.25, 0.50, 0.75, 0.99]]
    colors = ['#0077BB', '#EE7733', '#009988', '#CC3311']  # High-contrast scientific palette

    for i, idx in enumerate(t_indices):
        t_val = t_grid[idx]
        ax2.plot(x_grid, u_exact[idx, :], '-', color=colors[i], label=f'Exact t={t_val:.2f}', linewidth=2)
        ax2.plot(x_grid, u_pred[idx, :], '--', color=colors[i], label=f'c-PINN t={t_val:.2f}', linewidth=2)

    ax2.set_title("1D Velocity Profile Slices over Time (Exact vs c-PINN)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("u(x, t)", fontsize=12)
    ax2.legend(ncol=4, loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('evaluation/plots/fig1_exact_vs_cpinn.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: Pointwise Absolute Error Map
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(X, T, abs_error, cmap='YlOrRd', shading='auto')
    ax.set_title("c-PINN Pointwise Absolute Error |u_exact - u_pred|", fontsize=14, fontweight='bold')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.savefig('evaluation/plots/fig2_pointwise_error.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 3: XAI Layer 2 & 3 Hidden Activation Heatmaps (Epoch 0, 1000, 5000)
    # -------------------------------------------------------------------------
    epochs = [0, 1000, 5000]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, ep in enumerate(epochs):
        act2 = np.load(f'evaluation/epoch_{ep}/layer2_activations.npy')  # (25600, 40)
        act3 = np.load(f'evaluation/epoch_{ep}/layer3_activations.npy')  # (25600, 40)

        # Average activations across evaluation grid to form 2D neuron heatmap (40 neurons x 40 sub-grids)
        act2_reshape = act2[:10000, :40].reshape(100, 100, 40).mean(axis=0)  # (100, 40)
        act3_reshape = act3[:10000, :40].reshape(100, 100, 40).mean(axis=0)  # (100, 40)

        c2 = axes[0, col].imshow(act2_reshape, cmap='inferno', aspect='auto', vmin=-1, vmax=1)
        axes[0, col].set_title(f"Layer 2 Activations (Epoch {ep})", fontsize=12, fontweight='bold')
        axes[0, col].set_xlabel("Neuron Index (0-39)")
        axes[0, col].set_ylabel("Grid Position")

        c3 = axes[1, col].imshow(act3_reshape, cmap='inferno', aspect='auto', vmin=-1, vmax=1)
        axes[1, col].set_title(f"Layer 3 Activations (Epoch {ep})", fontsize=12, fontweight='bold')
        axes[1, col].set_xlabel("Neuron Index (0-39)")
        axes[1, col].set_ylabel("Grid Position")

    plt.suptitle("XAI Feature Tracking: Classical Layer Hidden Activation Heatmaps", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('evaluation/plots/fig3_xai_activations.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 4: NTK Heatmaps & Eigenvalue Spectrum Decay
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    eigenvalues_list = []
    ntk_colors = ['#2B5C8F', '#D95F02', '#7570B3']
    for idx, ep in enumerate(epochs):
        K = np.load(f'evaluation/epoch_{ep}/ntk_K.npy')  # (1024, 1024)
        K_norm = K / np.trace(K)

        c = axes[idx].imshow(K_norm[:100, :100], cmap='cividis')
        axes[idx].set_title(f"NTK Matrix K (Epoch {ep})", fontsize=12, fontweight='bold')
        fig.colorbar(c, ax=axes[idx])

        # Compute eigenvalues
        evals = np.linalg.eigvalsh(K)
        evals = np.sort(evals)[::-1]
        eigenvalues_list.append((ep, evals))

    # Eigenspectrum decay
    for idx, (ep, evals) in enumerate(eigenvalues_list):
        axes[3].semilogy(evals[:100], label=f'Epoch {ep}', linewidth=2, color=ntk_colors[idx])

    axes[3].set_title("NTK Eigenspectrum Decay", fontsize=12, fontweight='bold')
    axes[3].set_xlabel("Eigenvalue Rank", fontsize=11)
    axes[3].set_ylabel("Eigenvalue (log scale)", fontsize=11)
    axes[3].legend()
    axes[3].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('evaluation/plots/fig4_ntk_spectrum.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 5: Training Loss Convergence & Rel L2 History
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.semilogy(history['epoch'], history['loss_total'], label='Total Loss', color='#2C3E50', linewidth=2)
    ax1.semilogy(history['epoch'], history['loss_f'], label='Physics Residual (MSE_f)', color='#E74C3C', linestyle='--')
    ax1.semilogy(history['epoch'], history['loss_i'], label='Initial Loss (MSE_i)', color='#2980B9', linestyle='--')
    ax1.semilogy(history['epoch'], history['loss_b'], label='Boundary Loss (MSE_b)', color='#27AE60', linestyle='--')
    ax1.set_title("c-PINN Loss Convergence Curves", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (log scale)", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.semilogy(history['epoch'], history['relative_l2'], color='#8E44AD', linewidth=2)
    ax2.set_title("Relative L2 Validation Error Floor", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Relative L2 Error", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('evaluation/plots/fig5_loss_convergence.png', dpi=300)
    plt.close()

    print("Successfully generated all evaluation figures in evaluation/plots/!")

if __name__ == '__main__':
    evaluate_and_plot()
