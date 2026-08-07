import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pennylane as qml

from src.vqc_circuit import create_vqc_node

class QuantumDiagnostics:
    """
    Quantum Information Theory Diagnostics for Hybrid QAPINNs.
    
    Computes:
    1. Von Neumann Entropy S(rho_A) of reduced density matrices to quantify entanglement.
    2. Quantum Parameter Gradient Variance Var(dL/dtheta_q) to monitor for Barren Plateaus.
    3. Post-quantum activation maps across the evaluation coordinate grid.
    """
    def __init__(self, model):
        self.model = model

    def compute_von_neumann_entropy(self, X_eval_sample):
        """
        Computes average Von Neumann Entropy S(rho_A) across sample coordinate points.
        rho_A is obtained by tracing out half of the qubits.
        """
        self.model.eval()
        n_qubits = self.model.n_qubits
        n_layers = self.model.n_layers
        
        # State vector QNode
        state_qnode = create_vqc_node(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding=self.model.encoding,
            measurement_mode='state'
        )

        # Get weights from model's qlayer
        weights = self.model.qlayer.weights.detach()

        entropies = []
        num_samples = min(50, X_eval_sample.shape[0])

        for i in range(num_samples):
            x_single = X_eval_sample[i:i+1]
            if self.model.position == 'middle':
                with torch.no_grad():
                    h_in = self.model.act(self.model.fc_in(x_single))
                    x_single = self.model.fc_pre_vqc(h_in)
            elif self.model.position == 'final':
                with torch.no_grad():
                    h1 = self.model.act(self.model.fc1(x_single))
                    h2 = self.model.act(self.model.fc2(h1))
                    x_single = self.model.fc_pre_vqc(h2)

            state = state_qnode(x_single.squeeze(0), weights)
            psi = state.detach().cpu().numpy()

            # Reshape into matrix for bipartition (first n_qubits//2 vs remaining)
            n_A = n_qubits // 2
            dim_A = 2**n_A
            dim_B = 2**(n_qubits - n_A)
            psi_mat = psi.reshape((dim_A, dim_B))

            # Reduced density matrix rho_A = psi_mat @ psi_mat.conj().T
            rho_A = psi_mat @ psi_mat.conj().T
            
            # Eigenvalues of rho_A
            eigenvals = np.linalg.eigvalsh(rho_A)
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zero noise
            
            # S(rho_A) = - sum (lambda * log2(lambda))
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            entropies.append(entropy)

        return float(np.mean(entropies))

    def compute_quantum_gradient_variance(self, loss_fn_eval):
        """
        Computes variance of gradients with respect to quantum parameters Var(dL/dtheta_q).
        """
        self.model.zero_grad()
        loss = loss_fn_eval()
        loss.backward(retain_graph=True)

        quantum_grads = []
        for name, p in self.model.named_parameters():
            if 'qlayer' in name and p.grad is not None:
                quantum_grads.append(p.grad.view(-1).detach().cpu().numpy())

        if len(quantum_grads) > 0:
            all_q_grads = np.concatenate(quantum_grads)
            var_q_grad = float(np.var(all_q_grads))
            mean_abs_q_grad = float(np.mean(np.abs(all_q_grads)))
        else:
            var_q_grad = 0.0
            mean_abs_q_grad = 0.0

        return var_q_grad, mean_abs_q_grad

    def capture_post_quantum_activations(self, X_eval):
        """Extracts activation matrix after the quantum layer across evaluation grid."""
        self.model.eval()
        with torch.no_grad():
            if self.model.position == 'first':
                q_out = self.model.qlayer(X_eval)
            elif self.model.position == 'middle':
                h_in = self.model.act(self.model.fc_in(X_eval))
                pre_vqc = self.model.fc_pre_vqc(h_in)
                q_out = self.model.qlayer(pre_vqc)
            elif self.model.position == 'final':
                h1 = self.model.act(self.model.fc1(X_eval))
                h2 = self.model.act(self.model.fc2(h1))
                pre_vqc = self.model.fc_pre_vqc(h2)
                q_out = self.model.qlayer(pre_vqc)
        return q_out.cpu().numpy()

if __name__ == '__main__':
    from src.qapinn import HybridQAPINN

    print("Testing Quantum Diagnostics Module...")
    model = HybridQAPINN(n_qubits=4, position='first')
    diag = QuantumDiagnostics(model)

    X_sample = torch.randn(50, 2)
    entropy = diag.compute_von_neumann_entropy(X_sample)
    print(f"Initial Von Neumann Entanglement Entropy S(rho_A): {entropy:.5f}")

    def dummy_loss():
        u = model(X_sample)
        return torch.mean(u**2)

    var_grad, mean_grad = diag.compute_quantum_gradient_variance(dummy_loss)
    print(f"Quantum Parameter Gradient Variance: {var_grad:.5e}, Mean Abs Grad: {mean_grad:.5e}")
    print("Barren Plateau Check:", "HEALTHY (No Barren Plateau)" if var_grad > 1e-7 else "BARREN PLATEAU DETECTED")
