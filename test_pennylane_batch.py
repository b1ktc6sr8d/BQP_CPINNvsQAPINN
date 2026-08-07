# pyrefly: ignore [missing-import]
import pennylane as qml
# pyrefly: ignore [missing-import]
import torch

n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def circuit(inputs, weights):
    for i in range(n_qubits):
        if i % 2 == 0:
            qml.RX(inputs[..., 0] * 3.14159, wires=i)
        else:
            qml.RY(inputs[..., 1] * 3.14159, wires=i)
            
    for l in range(2):
        for q in range(n_qubits):
            qml.RX(weights[l, q, 0], wires=q)
            qml.RY(weights[l, q, 1], wires=q)
            qml.RZ(weights[l, q, 2], wires=q)
            
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qlayer = qml.qnn.TorchLayer(circuit, {'weights': (2, 4, 3)})
x = torch.randn(10, 2)
out = qlayer(x)
print("Output shape:", out.shape)
