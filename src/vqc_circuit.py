# pyrefly: ignore [missing-import]
import pennylane as qml
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np


# =============================================================================
# BACKEND AVAILABILITY CHECK
# =============================================================================

def check_backend(device_name: str) -> str:
    """
    Validates that the requested PennyLane device is available on this machine.
    Falls back to 'default.qubit' with a printed warning if the plugin is missing.

    Supported devices:
        'default.qubit'   -- Always available (pure Python/NumPy simulator)
        'lightning.qubit' -- Requires: pip install pennylane-lightning
        'lightning.gpu'   -- Requires: pip install pennylane-lightning[gpu]
                             AND:      NVIDIA GPU + CUDA 11.x or 12.x
    """
    try:
        qml.device(device_name, wires=1)   # quick probe
        return device_name
    except Exception:
        print(f"\n  [!] Device '{device_name}' is NOT available on this machine.")
        print( "      Falling back to 'default.qubit' automatically.")
        print( "      See the installation guide below:\n")
        _print_install_guide(device_name)
        return 'default.qubit'


def _print_install_guide(device_name: str):
    """Prints the pip installation steps for the missing backend."""
    if device_name == 'lightning.qubit':
        print("  +---------------------------------------------------------+")
        print("  |  INSTALL  lightning.qubit  (CPU C++ backend)            |")
        print("  |                                                          |")
        print("  |  pip install pennylane-lightning                         |")
        print("  |                                                          |")
        print("  |  No GPU required.  5-10x faster than default.qubit.     |")
        print("  +---------------------------------------------------------+")
    elif device_name == 'lightning.gpu':
        print("  +---------------------------------------------------------+")
        print("  |  INSTALL  lightning.gpu  (NVIDIA GPU backend)           |")
        print("  |                                                          |")
        print("  |  Step 1 — Install CUDA Toolkit 11.x or 12.x             |")
        print("  |    https://developer.nvidia.com/cuda-downloads           |")
        print("  |                                                          |")
        print("  |  Step 2 — Install cuQuantum                              |")
        print("  |    pip install cuquantum-python                          |")
        print("  |                                                          |")
        print("  |  Step 3 — Install PennyLane lightning-gpu plugin         |")
        print("  |    pip install pennylane-lightning[gpu]                  |")
        print("  |                                                          |")
        print("  |  Step 4 — Verify GPU is detected                         |")
        print("  |    python -c \"import pennylane as qml; qml.about()\"      |")
        print("  |                                                          |")
        print("  |  Requirements:                                           |")
        print("  |    NVIDIA GPU  (Volta / Turing / Ampere / Ada Lovelace)  |")
        print("  |    CUDA 11.x or 12.x                                    |")
        print("  |    Windows 10 / 11  64-bit                               |")
        print("  +---------------------------------------------------------+")
    print()


# =============================================================================
# DIFF METHOD SELECTOR
# =============================================================================

def _get_diff_method(device_name: str) -> str:
    """
    Returns the best gradient differentiation method for the given backend.

        default.qubit   -> 'backprop'
            PyTorch autograd traces directly through the NumPy simulator.

        lightning.qubit -> 'adjoint'
            The adjoint differentiation method reuses the state vector to
            compute all gradients in a single reverse pass.
            5-10x faster than backprop on CPU.

        lightning.gpu   -> 'adjoint'
            Same adjoint method, executed on NVIDIA GPU via CUDA.
            20-50x faster than default.qubit.
    """
    if device_name == 'default.qubit':
        return 'backprop'
    elif device_name in ('lightning.qubit', 'lightning.gpu'):
        return 'adjoint'
    else:
        return 'best'


# =============================================================================
# VQC NODE FACTORY
# =============================================================================

def create_vqc_node(n_qubits=4, n_layers=2, encoding='angle',
                    entanglement='linear', measurement_mode='expectation',
                    device_name='lightning.qubit'):
    """
    Creates and returns a PennyLane QNode (quantum circuit function)
    on the specified simulator backend.

    Parameters:
        n_qubits (int)         : Number of qubits — 3, 4, or 5
        n_layers (int)         : Number of variational layers
        encoding (str)         : 'angle' or 'amplitude'
        entanglement (str)     : 'linear' (chain) or 'circular' (ring)
        measurement_mode (str) : 'expectation' | 'probability' | 'state'
        device_name (str)      : PennyLane backend:
                                   'default.qubit'    always available
                                   'lightning.qubit'  C++ CPU  (5-10x faster)
                                   'lightning.gpu'    NVIDIA GPU (20-50x faster)
    """
    # Validate device and pick matching diff method
    device_name = check_backend(device_name)
    diff_method = _get_diff_method(device_name)

    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):

        # ── 1. DATA ENCODING LAYER ──────────────────────────────────────────
        if encoding == 'angle':
            # Angle encoding: map x -> RX on even qubits, t -> RY on odd qubits
            for i in range(n_qubits):
                if i % 2 == 0:
                    qml.RX(inputs[..., 0] * np.pi, wires=i)
                else:
                    qml.RY(inputs[..., 1] * np.pi, wires=i)

        elif encoding == 'amplitude':
            norm_factor = torch.sqrt(
                inputs[..., 0] ** 2 + inputs[..., 1] ** 2 + 1e-8)
            for i in range(n_qubits):
                qml.RY(np.pi * inputs[..., 0] / norm_factor, wires=i)
                qml.RZ(np.pi * inputs[..., 1] / norm_factor, wires=i)

        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # ── 2. VARIATIONAL QUANTUM LAYERS ───────────────────────────────────
        for l in range(n_layers):
            # Parameterized single-qubit rotations (learnable weights)
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)

            # Entanglement topology
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])            # linear chain
            if entanglement == 'circular' and n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])     # close the ring

        # ── 3. MEASUREMENT INTERFACING ──────────────────────────────────────
        if measurement_mode == 'expectation':
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        elif measurement_mode == 'probability':
            return qml.probs(wires=range(n_qubits))
        elif measurement_mode == 'state':
            return qml.state()
        else:
            raise ValueError(f"Unknown measurement_mode: {measurement_mode}")

    return circuit


def get_weight_shape(n_qubits=4, n_layers=2):
    """Returns the weight shape dict required by PennyLane TorchLayer."""
    return {"weights": (n_layers, n_qubits, 3)}


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing VQC with backend selection ...\n")

    test_inputs  = torch.randn(10, 2, dtype=torch.float32)

    for dev_name in ['default.qubit', 'lightning.qubit', 'lightning.gpu']:
        resolved = check_backend(dev_name)
        circuit  = create_vqc_node(n_qubits=4, n_layers=2,
                                   measurement_mode='expectation',
                                   device_name=dev_name)
        qlayer = qml.qnn.TorchLayer(circuit, get_weight_shape(4, 2))
        out    = qlayer(test_inputs)
        print(f"  [{resolved:20s}]  Output shape: {out.shape}  OK")

    print("\nAll backend tests done.")
