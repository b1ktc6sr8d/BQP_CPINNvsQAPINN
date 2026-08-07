# WISER Summer Program 2026: Classical PINN vs Quantum-Assisted PINN (QAPINN) for CFD

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-EE4C2C.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.42.3-purple.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end computational fluid dynamics (CFD) research project developed for the **WISER Global Quantum+AI Program 2026** (in collaboration with **BQP**). 

This project investigates the **Explainability and Learning Dynamics of Quantum Layers in Physics-Informed Neural Networks (PINNs)** by comparing a Classical PINN (c-PINN) against a Hybrid Quantum-Assisted PINN (QAPINN) solving the 1D Viscous Burgers' Equation.

---

## 🌟 Benchmarking Highlights & Key Results

| Metric / Benchmark | Classical Baseline (c-PINN) | Hybrid QAPINN (Position 1, 4 Qubits) | Improvement / Impact |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters ($P$)** | **3,441** | **1,905** | **44.64% Parameter Reduction** |
| **Relative $L_2$ Error Floor** | **0.19486** (19.48%) | **0.20227** (20.22%) | Equivalent precision with ~half the parameters |
| **XAI Feature Representation** | Rigid, uniform linear bands | Smooth, non-linear quantum state maps | Breaks classical feature bottleneck |
| **Entanglement Entropy $S(\rho_A)$** | N/A (Classical) | **0.66 $\to$ 0.93 bits** | Verifies quantum state entanglement growth |
| **Barren Plateau Monitor** | N/A | **$\text{Var} > 1.1 \times 10^{-5} \gg 10^{-7}$** | Healthy active gradient flow confirmed |

---

## 📖 Simple Concept Explanations (No Math Degree Needed!)

### 1. What is the 1D Viscous Burgers' Equation?
The 1D viscous Burgers' Equation is a simple non-linear partial differential fluid equation that shows how a wave moves("Advection") and smooths out at the same time("Diffusion").

In math form:

```text
u u_{xx} + u u_x + u_t = 0
```

This means:
- `u(x,t)` is the velocity of fluid at position `x` and time `t`.
- `u_t` is how the velocity changes over time.
- `u_x` is how the velocity changes in space.
- `u_{xx}` is how sharply the velocity changes in space (diffusion).
- `\nu` is viscosity, a number that controls how much the fluid smooths itself.

In our project, `\nu = 0.01`, so there is a visible balance between:
- a wave getting steeper because the fluid pushes itself (`u u_x`),
- and the fluid smoothing itself because of viscosity (`\nu u_{xx}`).

Example: imagine a simple hill of water moving and slowly turning into a sharp front.

### 1.1 Why Burgers' Equation is important
- It is a simplified model for compressible fluid flow and traffic flow.
- It demonstrates both sharp shock forming and how viscosity smooths it.
- It is a canonical Benchmark for Physics-informed Learning because it has known analytical solution under specific initial/boundary conditions. It is easy to compare with a known exact solution, so it is great for testing AI methods.

### 1.2 Typical initial/boundary conditions in this project
We often start with a sine wave shape:

```text
u(x,0) = -\sin(\pi x)
```

and fix the edges to zero:

```text
u(a,t) = u(b,t) = 0
```

So the wave starts smooth, then evolves into a steep profile near the center.

### 2. What is the Cole-Hopf Transformation?
The Cole-Hopf transformation is a clever trick that turns Burgers' nonlinear equation into a simple linear heat equation.

We define a helper function `\phi(x,t)` and then compute:

```text
u(x,t) = -2 \nu \frac{\partial}{\partial x} \log \phi(x,t)
```

After this change, `\phi` satisfies:

```text
\phi_t = \nu \phi_{xx}
```

This is much easier to solve, so we can find the exact answer for `u(x,t)` and use it as a true reference.

### 3. What is a PINN (Physics-Informed Neural Network)?
A PINN is a neural network that learns a solution to a physics equation by checking the equation while it trains.

Instead of only using examples, it also learns from the physics law itself.

Example use cases:
- fluid flow and heat transfer,
- wave propagation in materials,
- solving equations for air pressure or temperature.

A PINN learns `\hat{u}(x,t)` and minimizes a combined loss:

```text
\mathcal{L} = \text{MSE}_f + \text{MSE}_i + \text{MSE}_b
```

where:
- `\text{MSE}_f` (physical residual loss from Burgers' eqn.)measures how well the network satisfies the equation,
- `\text{MSE}_i` (initial condition loss)measures how well it matches the initial condition,
- `\text{MSE}_b` (boundary condition loss)measures how well it matches the boundary values.

The physics residual is:

```text
f(x,t) = u_t + u u_x - \nu u_{xx}
```

A good PINN makes `f(x,t)` close to zero everywhere.

How c-PINN works in this project:
1. Take `x` and `t` as inputs.
2. Pass them through several classical neural network layers.
3. Output a velocity prediction `\hat{u}(x,t)`.
4. Compute the physics residual and condition errors.
5. Train the network so both the prediction and the physics match.

### 4. What is a QAPINN (Quantum-Assisted PINN)?
A QAPINN is a hybrid model that mixes classical neural network layers with a quantum layer built from a Variational Quantum Circuit(VQE).

It works like this:
1. Take `(x,t)` inputs.
2. Process them with some classical layers.
3. Send part of the result into a quantum circuit.
4. Measure the quantum circuit and return a feature vector.
5. Continue with classical layers to produce `\hat{u}(x,t)`.

Why use QAPINN?
- It can learn richer patterns with fewer parameters.
- It can create nonlinear features that are hard for a normal neural network.
- It is useful when quantum circuits can add expressiveness without needing very many qubits.

Example use cases:
- small physics problems where quantum features can improve accuracy,
- research in quantum machine learning and hybrid models,
- testing whether quantum layers help classical physics-based learning.

### 5. What is a VQC (Variational Quantum Circuit)?
A VQC is a quantum circuit with adjustable gates.

It has two main parts:
- **Data encoding:** put the input values into the circuit by rotating qubits,
- **Variational layers:** apply trainable rotation and entangling gates.

Common gates in a VQC:
- `R_x(\theta)`: rotate a qubit around the x-axis,
- `R_y(\theta)`: rotate a qubit around the y-axis,
- `R_z(\theta)`: rotate a qubit around the z-axis,
- `CNOT`: entangle two qubits so their states depend on each other.

Example circuit steps:
1. encode `x` and `t` with `R_y(x)` and `R_z(t)`,
2. apply `R_x(\theta_1)`, `R_y(\theta_2)` on each qubit,
3. add `CNOT` gates between qubits,
4. measure the circuit to produce outputs.

Why it is called "variational": because the gates have parameters `\theta` that are adjusted during training, just like weights in a neural network.

### 6. What is Von Neumann Entanglement Entropy (`S(\rho_A)`)?
Von Neumann Entropy measures the entanglement in a Quantum State. In a quantum circuit, let `\rho` be the density matrix, which describes the full quantum state.

When we split the qubits into two groups, `\rho_A` is the density matrix for just one group.

The entropy is:

```text
S(\rho_A) = - \mathrm{Tr}(\rho_A \log \rho_A)
```

What this means:
- `\rho` describes probabilities and quantum interference in the circuit,
- `\rho_A` describes the state of one part of the circuit after ignoring the other part,
- `S(\rho_A)` measures how much the two parts are entangled.

If entropy rises:
- the qubits are more strongly connected,
- the quantum circuit is using entanglement as a feature.

If entropy falls:
- the circuit is becoming more separable,
- the quantum part may be losing power to represent complex correlations.

Example: if qubits are independent, `S(\rho_A)=0`; if they are strongly entangled, `S(\rho_A)` is larger.

### 7. What is a Barren Plateau?
A barren plateau is when the quantum circuit has almost zero gradient for all parameters.

That means training cannot improve the circuit because the loss stops changing.

We monitor this by checking the variance of quantum gradients:

```text
\mathrm{Var}\left(\frac{\partial \mathcal{L}}{\partial \theta_q}\right)
```

If the variance is too small, the quantum layer is stuck.
If it stays large enough, the quantum layer is still learning.

Simple analogy: a barren plateau is like walking on a flat desert where you cannot tell which way is uphill.

---

## 📁 Repository Architecture

```
BQP_CPINNvsQAPINN/
├── src/
│   ├── domain.py               # Spatiotemporal domain sampler (N_f=2000, N_i=200, N_b=200)
│   ├── exact_solution.py       # Cole-Hopf exact reference solution calculator
│   ├── cpinn.py                # Classical 4-layer c-PINN PyTorch model (P=3,441)
│   ├── xai_hooks.py            # Classical activation & NTK tracking hooks
│   ├── vqc_circuit.py          # Parameterized VQC in PennyLane (default.qubit)
│   ├── qapinn.py               # Hybrid QAPINN PyTorch framework (P=1,905)
│   └── quantum_diagnostics.py  # Von Neumann Entropy & Barren Plateau gradient monitor
├── train_cpinn.py              # Part 1: Trains c-PINN for 5,000 epochs & saves baseline XAI
├── evaluate_cpinn.py           # Part 1: Computes classical baseline metrics & Figures 1-5
├── train_qapinn.py             # Part 2: Trains Hybrid QAPINN model & quantum diagnostics
└── evaluate_qapinn.py          # Part 2: Generates comparative metrics & Figures 6-9
```

---

## 🚀 Step-by-Step Installation & Execution Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/b1ktc6sr8d/BQP_CPINNvsQAPINN.git
cd bqp_cpinnvsqapinn
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install torch numpy scipy matplotlib pennylane
```

---

### Step 3: Run Part 1 (Classical PINN Baseline)

1. **Calculate Cole-Hopf Exact Ground Truth & Train c-PINN**:
   ```bash
   python train_cpinn.py
   ```
   *Trains the 4-layer c-PINN ($P=3,441$) over 5,000 epochs and saves XAI tracking snapshots at Epochs 0, 1000, 5000.*

2. **Generate Baseline Metrics & Figures 1–5**:
   ```bash
   python evaluate_cpinn.py
   ```
   *Generates exact vs predicted shock wave maps, 1D time slices, pointwise error heatmaps, classical activation heatmaps, and NTK eigenspectrum plots inside `evaluation/plots/`.*

---

### Step 4: Run Part 2 (Hybrid QAPINN & Quantum Integration)

1. **Train Hybrid QAPINN Model & Compute Quantum Diagnostics**:
   ```bash
   python train_qapinn.py
   ```
   *Trains the 4-qubit Hybrid QAPINN ($P=1,905$) over 5,000 epochs, tracks Von Neumann Entanglement Entropy, and monitors for Barren Plateaus.*

2. **Generate Comparative Figures 6–9**:
   ```bash
   python evaluate_qapinn.py
   ```
   *Generates comparative solution maps, quantum vs classical XAI heatmaps, entanglement growth curves, and parameter compression Pareto charts inside `evaluation/plots/`.*

*(Optional) Run full multi-configuration benchmark suite across 6 model variants:*
```bash
python train_qapinn.py --all
```

---

## 📊 Figure Explanations & Visual Analysis

### Figure 1: Exact Analytical Solution vs c-PINN Prediction & 1D Profiles
- **Top Left**: True Cole-Hopf fluid velocity field showing initial sine wave steepening into a shock wave.
- **Top Right**: c-PINN prediction field ($P=3,441$).
- **Bottom**: 1D velocity profile slices at $t=0.25, 0.50, 0.75, 0.99$ matching exact analytical curves.

### Figure 2: Pointwise Absolute Error Map $|u_{\text{true}} - \hat{u}|$
- Displays absolute prediction error across space and time. High error is localized strictly at the sharp shock front ($x \approx 0.5, t > 0.5$).

### Figure 3: Classical Hidden Layer Activation Heatmaps (XAI Baseline)
- Visualizes Layer 2 and Layer 3 neuron activations. Shows that classical hidden layers form **rigid, uniform, parallel linear stripes**.

### Figure 4: Neural Tangent Kernel (NTK) & Eigenspectrum Decay
- Displays NTK matrices $\mathbf{K} = \mathbf{J}\mathbf{J}^T$ and eigenvalue decay curves showing fast convergence on smooth features vs slow convergence on sharp shock details.

### Figure 5: Training Loss Convergence & Relative $L_2$ History
- Log-scale convergence of Total Loss, Physics Residual $\text{MSE}_f$, Initial Loss $\text{MSE}_i$, Boundary Loss $\text{MSE}_b$, and Relative $L_2$ error floor ($0.19486$).

---

### Figure 6: Comparative Fluid Flow Predictions (Exact vs c-PINN vs QAPINN)
- Compares Cole-Hopf exact solution vs Classical c-PINN ($P=3,441$) vs Hybrid QAPINN ($P=1,905$).
- Proves that the **1,905-parameter QAPINN** matches the 3,441-parameter classical PINN solution field across space and time!

### Figure 7: Quantum vs Classical XAI Activation Heatmaps
- Compares classical linear band heatmaps against post-quantum state feature maps.
- Shows that the 4-qubit VQC breaks classical rigid linear stripes, creating a smooth, non-linear quantum feature distribution.

### Figure 8: Quantum Information Diagnostics (Entanglement Entropy & Barren Plateau Check)
- **Left**: Tracks **Von Neumann Entanglement Entropy** $S(\rho_A)$ rising from $0.66$ to $>0.93$ bits over 5,000 epochs.
- **Right**: Monitors quantum parameter gradient variance $\text{Var}(\partial \mathcal{L}/\partial \theta_q)$, confirming active gradient flow above the $10^{-7}$ Barren Plateau threshold.

### Figure 9: Parameter Compression vs Accuracy Pareto Trade-Off
- Bar chart illustrating **44.64% parameter reduction** ($3,441 \to 1,905$) while preserving baseline fluid simulation accuracy ($L_2 \approx 0.1948 \text{ vs } 0.2022$).

---

## 👥 Authors & Acknowledgments

- **WISER Global Quantum+AI Program 2026**
- **BQP (BosonQ Psi)**
- **GitHub Repository**: https://github.com/b1ktc6sr8d/BQP_CPINNvsQAPINN.git
