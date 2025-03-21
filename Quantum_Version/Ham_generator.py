from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qcels import ham_shift as scale_factor
import numpy as np
import subprocess
import os

def generate_TFIM_gates(qubits, steps, dt, g, location):
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    steps = steps - 1 
    if not os.path.exists("TFIM_Operators"):
        os.mkdir("TFIM_Operators")
    
    with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
        f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
        f.write("[Trotter]\nsteps = 1\ndt = 0\n\n")
        f.write("[Jy]\nvalue = 0\n\n")
        f.write("[Jz]\nvalue = 1\n\n")
        f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
        f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt=0_i=\nimin = 1\nimax = 2\nstep = 1\n")
    subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"])
    os.remove("TFIM_Operators/Operator_Generator.ini")
    gates = []
    qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt=0_i=1.qasm")
    gates.append(qc.to_gate(label = "TFIM 1").control())
    os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt=0_i=1.qasm")

    with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
        f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
        f.write("[Trotter]\nsteps = "+str(steps)+"\ndt = "+str(dt)+"\n\n")
        f.write("[Jy]\nvalue = 0\n\n")
        f.write("[Jz]\nvalue = 1\n\n")
        f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
        f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=\nimin = 1\nimax = "+str(steps)+"\nstep = 1\n")
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"])
    os.remove("TFIM_Operators/Operator_Generator.ini")
    for step in range(steps):
        qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step+1)+".qasm")
        gates.append(qc.to_gate(label = "TFIM "+str(step+2)).control())
        os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step+1)+".qasm")
    os.rmdir("TFIM_Operators")
    return gates

def create_hamiltonian(qubits, system, g=0, J=4, show_steps=False):
    assert(system.upper() == "TFIM" or system.upper() == "SPIN")
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
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
    H = scale_factor*H / np.linalg.norm(H, ord=2)
    
    # rotate matrix so that it will be positive definite (not nessary in this usecase)
    # H += np.pi*np.eye(2**qubits)

    if show_steps:
        eigenvalues = np.linalg.eigvals(H)
        print("Scaled eigenvalues of the hamiltonian:\n", np.linalg.eigvals(H))
        min_eigenvalue = np.min(eigenvalues)
        print("Lowest energy eigenvalue", min_eigenvalue); print()
    return H