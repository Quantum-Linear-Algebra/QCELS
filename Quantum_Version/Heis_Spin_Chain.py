from qiskit.quantum_info import Pauli
import numpy as np

def create_hamiltonian(qubits, system, g=0, J=4, show_steps=False):
    assert(system.upper() == "TFIM" or system.upper() == "SPIN")
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    scale_factor = (3/4)*np.pi
    if system.upper() == "TFIM":
        # construct the Hamiltonian
        # with Pauli Operators in Qiskit ^ represents a tensor product
        if show_steps: print("H = ", end='')
        for i in range(qubits-1):
            temp = Pauli('')
            for j in range(qubits):
                if (j == i or j == i+1):
                    temp ^= Pauli('Z')
                else:
                    temp ^= Pauli('I')
            H += -temp.to_matrix()
            if show_steps: print("-"+str(temp)+" ", end='')
        for i in range(qubits):
            temp = Pauli('')
            for j in range(qubits):
                if (j == i):
                    temp ^= Pauli('X')
                else:
                    temp ^= Pauli('I')
            H += -g*temp.to_matrix()
            if show_steps: print("-"+str(g)+"*"+str(temp)+" ", end='')
        if show_steps: print("\n")
        
    elif system.upper() == "SPIN":
        def S(index, coupling):
            temp = Pauli('')
            for j in range(qubits):
                if j == index:
                    temp ^= Pauli(coupling)
                else:
                    temp ^= Pauli('I')
            return 1/2*temp.to_matrix()
        if show_steps: print("H = ", end='\n')
        for qubit in range(qubits-1):
            H += S(qubit, 'X')@S(qubit+1, 'X')
            H += S(qubit, 'Y')@S(qubit+1, 'Y')
            H += S(qubit, 'Z')@S(qubit+1, 'Z')
        H += S(qubits-1, 'X')@S(0, 'X')
        H += S(qubits-1, 'Y')@S(0, 'Y')
        H += S(qubits-1, 'Z')@S(0, 'Z')
        H *= J
        if show_steps: print(H)

    if show_steps:
        eigenvalues = np.linalg.eigvals(H)
        print("Original eigenvalues:", eigenvalues)
    
    # scale eigenvalues of the Hamiltonian
    if J == 1:
        H = H
    else:
        H = scale_factor*H / np.linalg.norm(H, ord=2)
    
    # rotate matrix so that it will be positive definite (not nessary in this usecase)
    # H += np.pi*np.eye(2**qubits)

    if show_steps:
        eigenvalues = np.linalg.eigvals(H)
        print("Scaled eigenvalues of the hamiltonian:\n", np.linalg.eigvals(H))
        min_eigenvalue = np.min(eigenvalues)
        print("Lowest energy eigenvalue", min_eigenvalue); print()
    return H
