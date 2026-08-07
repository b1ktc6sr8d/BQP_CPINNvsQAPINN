# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import numpy as np
import os

class XAITracker:
    """
    Explainable AI (XAI) tracking infrastructure for extracting hidden neuron activations,
    empirical Parameter Jacobian Matrix (J), and Neural Tangent Kernel (K = J J^T).
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks at Layer 2 and Layer 3 linear activations."""
        # In Sequential: 0:Linear(2,40), 1:Tanh, 2:Linear(40,40), 3:Tanh, 4:Linear(40,40), 5:Tanh, 6:Linear(40,1)
        layer2_linear = self.model.net[2]
        layer3_linear = self.model.net[4]

        def get_activation(layer_name):
            def hook(module, input, output):
                self.activations[layer_name] = output.detach().cpu().numpy()
            return hook

        self.hooks.append(layer2_linear.register_forward_hook(get_activation('layer2')))
        self.hooks.append(layer3_linear.register_forward_hook(get_activation('layer3')))

    def capture_activations(self, X_eval):
        """Runs forward pass on evaluation grid and returns extracted hidden layer activation arrays."""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(X_eval)
        return self.activations['layer2'], self.activations['layer3']

    def compute_jacobian_and_ntk(self, X_eval_sub):
        """
        Computes the empirical Parameter Jacobian Matrix J (N_eval x P)
        and Neural Tangent Kernel matrix K = J J^T (N_eval x N_eval).
        """
        self.model.eval()
        N_eval = X_eval_sub.shape[0]
        params = [p for p in self.model.parameters() if p.requires_grad]
        P = sum(p.numel() for p in params)

        J = np.zeros((N_eval, P), dtype=np.float32)

        for i in range(N_eval):
            self.model.zero_grad()
            x_single = X_eval_sub[i:i+1]
            u_single = self.model(x_single)
            u_single.backward()

            grads = []
            for p in params:
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                else:
                    grads.append(torch.zeros(p.numel(), device=X_eval_sub.device))
            J[i, :] = torch.cat(grads).detach().cpu().numpy()

        # Neural Tangent Kernel K = J @ J.T
        K = J @ J.T
        return J, K

    def save_snapshot(self, epoch, X_eval, X_eval_sub, save_dir='evaluation'):
        """Captures and saves activations, Jacobian J, and NTK K to disk for a given epoch."""
        epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        act2, act3 = self.capture_activations(X_eval)
        J, K = self.compute_jacobian_and_ntk(X_eval_sub)

        np.save(os.path.join(epoch_dir, 'layer2_activations.npy'), act2)
        np.save(os.path.join(epoch_dir, 'layer3_activations.npy'), act3)
        np.save(os.path.join(epoch_dir, 'jacobian_J.npy'), J)
        np.save(os.path.join(epoch_dir, 'ntk_K.npy'), K)

        print(f"Saved XAI tracking snapshot at Epoch {epoch} to {epoch_dir}")
        print(f"  Layer 2 Activations: {act2.shape}, Layer 3 Activations: {act3.shape}")
        print(f"  Jacobian J: {J.shape}, NTK K: {K.shape}")

    def remove_hooks(self):
        """Removes forward hooks."""
        for hook in self.hooks:
            hook.remove()

if __name__ == '__main__':
    from cpinn import ClassicalPINN
    from domain import BurgersDomain

    model = ClassicalPINN()
    domain = BurgersDomain()
    X_mesh, T_mesh, X_eval, _, _ = domain.generate_eval_grid(Nx=50, Nt=50)
    X_eval_sub = X_eval[:100]

    tracker = XAITracker(model)
    tracker.save_snapshot(epoch=0, X_eval=X_eval, X_eval_sub=X_eval_sub)
    tracker.remove_hooks()
