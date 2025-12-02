import torch
import pennylane as qml
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch_geometric.utils import to_dense_adj

n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)
        
def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def RY_RX_layer(weights):
    """Applies a layer of parametrized RY and RX rotations."""
    for i, w in enumerate(weights):
        qml.RY(w[0], wires=i)
        qml.RX(w[1], wires=i)

def full_entangling_layer(n_qubits):
    """Applies CNOT gates between all pairs of qubits."""
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                qml.CNOT(wires=[i, j])

def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOTs."""
    for i in range(nqubits - 1):
        qml.CRZ(np.pi / 2, wires=[i, i + 1])
    for i in range(0, nqubits - 1, 2):
        qml.SWAP(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.SWAP(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def quantum_circuit(q_input_features, q_weights_flat, q_depth, n_qubits):
    """The variational quantum circuit."""

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    #Start from state |+>, unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    qml.AngleEmbedding(q_input_features, wires=range(n_qubits), rotation="Z")

    # Sequence of trainable variational layers
    for k in range(q_depth):
        if k % 2 == 0:
            entangling_layer(n_qubits)
            RY_layer(q_weights[k])
        else:
            full_entangling_layer(n_qubits)
            RY_RX_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return tuple(exp_vals)

class DressedQuantumCircuit(nn.Module):
    """A quantum circuit wrapped as a PyTorch module."""

    def __init__(self, n_qubits, q_depth = 1, q_delta = 0.001):
        """Definition of the *dressed* layout"""
        print('n_qubits: ', n_qubits)
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))

    def forward(self,input_features):
        """Optimized forward pass to reduce runtime."""

        # Quantum Embedding (U(X))
        q_in = torch.tanh(input_features) * np.pi / 2.0

        # Preallocate output tensor
        batch_size = q_in.shape[0]
        q_out = torch.zeros(batch_size, self.n_qubits, device = q_in.device)

        # Vectorized execution
        for i, elem in enumerate(q_in):
            q_out_elem = torch.hstack(quantum_circuit(elem, self.q_params, self.q_depth, self.n_qubits)).float()
            q_out[i] = q_out_elem

        return q_out
