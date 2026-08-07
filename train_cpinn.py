# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
import numpy as np
import os
import time

from src.domain import BurgersDomain
from src.cpinn import ClassicalPINN, compute_composite_loss
from src.xai_hooks import XAITracker

def train_cpinn(epochs=5000, lr=1e-3, nu=0.01, seed=42, device='cpu'):
    print("==================================================================")
    print("Starting Part 1 Classical PINN (c-PINN) Training Pipeline...")
    print("==================================================================")
    
    # 1. Initialize Domain and Model
    domain = BurgersDomain(seed=seed)
    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data(N_f=2000, N_i=200, N_b=200, device=device)
    
    # Evaluation grid for XAI tracking & L2 calculation
    X_mesh, T_mesh, X_eval, x_flat, t_flat = domain.generate_eval_grid(Nx=256, Nt=100, device=device)
    X_eval_sub = X_eval[::25].to(device)  # 1024 evaluation points for NTK computation

    # Load exact reference solution for Relative L2 tracking
    if os.path.exists('evaluation/exact_solution.npy'):
        u_exact = np.load('evaluation/exact_solution.npy')
    else:
        from src.exact_solution import compute_exact_grid
        u_exact = compute_exact_grid(x_flat, t_flat, nu=nu)

    model = ClassicalPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Initialize XAI Tracker
    tracker = XAITracker(model)

    # Loss history tracking
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_f': [],
        'loss_i': [],
        'loss_b': [],
        'relative_l2': []
    }

    start_time = time.time()

    # Snapshot at Epoch 0
    tracker.save_snapshot(0, X_eval, X_eval_sub)

    weights = (1.0, 10.0, 10.0)  # Stronger weighting on Initial and Boundary Conditions

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss, loss_f, loss_i, loss_b = compute_composite_loss(
            model, X_f, x_f, t_f, X_i, u_i, X_b, u_b, nu=nu, weights=weights
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute Relative L2 validation error
        if epoch % 100 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                u_pred = model(X_eval).cpu().numpy().reshape(u_exact.shape)
                l2_err = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

            history['epoch'].append(epoch)
            history['loss_total'].append(loss.item())
            history['loss_f'].append(loss_f.item())
            history['loss_i'].append(loss_i.item())
            history['loss_b'].append(loss_b.item())
            history['relative_l2'].append(l2_err)

            if epoch % 500 == 0 or epoch == epochs:
                print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6e} | "
                      f"Physics Residual (MSE_f): {loss_f.item():.6e} | "
                      f"Initial (MSE_i): {loss_i.item():.6e} | Boundary (MSE_b): {loss_b.item():.6e} | "
                      f"Rel L2 Error: {l2_err:.4e}")

        # Capture XAI tracking snapshot at key epochs
        if epoch in [1000, 5000]:
            tracker.save_snapshot(epoch, X_eval, X_eval_sub)

    tracker.remove_hooks()
    elapsed_time = time.time() - start_time
    print("==================================================================")
    print(f"Training Complete in {elapsed_time:.2f} seconds!")
    print(f"Final Relative L2 Validation Error Floor: {history['relative_l2'][-1]:.4e}")
    print("==================================================================")

    # Save model checkpoint and history
    os.makedirs('evaluation', exist_ok=True)
    torch.save(model.state_dict(), 'evaluation/cpinn_model.pt')
    np.save('evaluation/loss_history.npy', history)
    
    # Save final predictions on eval grid
    model.eval()
    with torch.no_grad():
        u_final_pred = model(X_eval).cpu().numpy().reshape(u_exact.shape)
    np.save('evaluation/cpinn_predictions.npy', u_final_pred)

    return model, history

if __name__ == '__main__':
    train_cpinn(epochs=5000, lr=1e-3, nu=0.01)
