# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np

class BurgersDomain:
    """
    Spatiotemporal domain generator for 1D Viscous Burgers' Equation.
    Domain: x in [0.0, 1.0], t in [0.0, 1.0]
    """
    def __init__(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0, seed=42):
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def sample_training_data(self, N_f=2000, N_i=200, N_b=200, device='cpu'):
        """
        Samples domain interior (N_f), initial condition (N_i), and boundary condition (N_b) points.
        Returns PyTorch tensors ready for autograd training.
        """
        # 1. Domain Interior Points (N_f = 2000 random points uniform in [0, 1] x [0, 1])
        x_f = torch.distributions.Uniform(self.x_min, self.x_max).sample((N_f, 1)).to(device)
        t_f = torch.distributions.Uniform(self.t_min, self.t_max).sample((N_f, 1)).to(device)
        x_f.requires_grad_(True)
        t_f.requires_grad_(True)
        X_f = torch.cat([x_f, t_f], dim=1)

        # 2. Initial Boundary Points (N_i = 200 at t = 0, u(x, 0) = sin(pi * x))
        x_i = torch.linspace(self.x_min, self.x_max, N_i).unsqueeze(1).to(device)
        t_i = torch.full_like(x_i, self.t_min).to(device)
        X_i = torch.cat([x_i, t_i], dim=1)
        u_i = torch.sin(np.pi * x_i)

        # 3. Spatial Boundary Points (N_b = 200 points tracking edges x = 0 and x = 1)
        t_b_half = torch.distributions.Uniform(self.t_min, self.t_max).sample((N_b // 2, 1)).to(device)
        x_b_left = torch.full_like(t_b_half, self.x_min).to(device)
        X_b_left = torch.cat([x_b_left, t_b_half], dim=1)

        x_b_right = torch.full_like(t_b_half, self.x_max).to(device)
        X_b_right = torch.cat([x_b_right, t_b_half], dim=1)

        X_b = torch.cat([X_b_left, X_b_right], dim=0)
        u_b = torch.zeros((N_b, 1), device=device)

        return X_f, x_f, t_f, X_i, u_i, X_b, u_b

    def generate_eval_grid(self, Nx=256, Nt=100, device='cpu'):
        """
        Generates a dense evaluation coordinate mesh grid over [x_min, x_max] x [t_min, t_max].
        Returns meshgrid (X, T) and flattened tensor X_eval (Nx*Nt, 2).
        """
        x_flat = np.linspace(self.x_min, self.x_max, Nx)
        t_flat = np.linspace(self.t_min, self.t_max, Nt)
        X_mesh, T_mesh = np.meshgrid(x_flat, t_flat)

        x_tensor = torch.tensor(X_mesh.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        t_tensor = torch.tensor(T_mesh.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        X_eval = torch.cat([x_tensor, t_tensor], dim=1)

        return X_mesh, T_mesh, X_eval, x_flat, t_flat

if __name__ == '__main__':
    domain = BurgersDomain()
    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data()
    print("Interior points shape:", X_f.shape)
    print("Initial points shape:", X_i.shape, "Initial u range:", u_i.min().item(), "to", u_i.max().item())
    print("Boundary points shape:", X_b.shape, "Boundary u values all 0:", (u_b == 0).all().item())
    X_mesh, T_mesh, X_eval, x_flat, t_flat = domain.generate_eval_grid()
    print("Eval grid shape:", X_eval.shape)
