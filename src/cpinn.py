# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

class ClassicalPINN(nn.Module):
    """
    Classical Physics-Informed Neural Network (c-PINN) for 1D Viscous Burgers' Equation.
    Architecture: 2 inputs (x, t) -> 3 hidden layers (40 neurons each, Tanh) -> 1 output u_hat.
    """
    def __init__(self, in_features=2, hidden_features=40, out_features=1, num_layers=3):
        super(ClassicalPINN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier Normal Initialization for smooth optimization with Tanh activations."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_t):
        """Forward pass predicting physical field velocity u_hat."""
        return self.net(x_t)

def compute_pde_residual(model, x_f, t_f, nu=0.01):
    """
    Calculates exact PDE residual of 1D Viscous Burgers' Equation via automatic differentiation:
    R = u_t + u * u_x - nu * u_xx
    """
    X_f = torch.cat([x_f, t_f], dim=1)
    u = model(X_f)

    # First derivatives: u_t and u_x
    grads = torch.autograd.grad(
        outputs=u,
        inputs=[t_f, x_f],
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )
    u_t = grads[0]
    u_x = grads[1]

    # Second derivative: u_xx
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]

    # Residual R = u_t + u * u_x - nu * u_xx
    residual = u_t + u * u_x - nu * u_xx
    return residual, u

def compute_composite_loss(model, X_f, x_f, t_f, X_i, u_i, X_b, u_b, nu=0.01, weights=(1.0, 1.0, 1.0)):
    """
    Computes total training loss balancing interior physics residual, initial conditions,
    and Dirichlet boundary conditions.
    """
    w_f, w_i, w_b = weights
    
    # 1. Physics Residual Loss
    residual, _ = compute_pde_residual(model, x_f, t_f, nu=nu)
    loss_f = torch.mean(residual ** 2)

    # 2. Initial Condition Loss
    u_i_pred = model(X_i)
    loss_i = torch.mean((u_i_pred - u_i) ** 2)

    # 3. Boundary Condition Loss
    u_b_pred = model(X_b)
    loss_b = torch.mean((u_b_pred - u_b) ** 2)

    # Total Loss
    total_loss = w_f * loss_f + w_i * loss_i + w_b * loss_b
    return total_loss, loss_f, loss_i, loss_b

if __name__ == '__main__':
    model = ClassicalPINN()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Classical PINN initialized successfully!")
    print(f"Architecture: 2 -> 40 -> 40 -> 40 -> 1")
    print(f"Total Trainable Parameters: {num_params}")
    
    # Test autograd pass
    x_dummy = torch.randn(10, 1, requires_grad=True)
    t_dummy = torch.randn(10, 1, requires_grad=True)
    res, u_pred = compute_pde_residual(model, x_dummy, t_dummy)
    print(f"PDE residual computation test passed. Residual shape: {res.shape}")
