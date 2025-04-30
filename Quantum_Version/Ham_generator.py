from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator
from qcels import ham_shift as scale_factor
import numpy as np
from numpy.linalg import eigh
import subprocess
import os

def generate_TFIM_gates(qubits, steps, dt, g, scaling, location, trotter = 1):
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    
    # calculate new scaled parameters
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    for i in range(qubits-1):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i or j == i+1):
                temp ^= Pauli('Z')
            else:
                temp ^= Pauli('I')
        H += -temp.to_matrix()
    for i in range(qubits):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i):
                temp ^= Pauli('X')
            else:
                temp ^= Pauli('I')
        H += -g*temp.to_matrix()
    n = 2**qubits

    eigs, _ = eigh(H)
    largest_eig = eigs[-1]
    coupling = scaling/largest_eig
    g *= scaling/largest_eig

    # calculate scaled Hamiltonian
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(qubits-1):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i or j == i+1):
                temp ^= Pauli('Z')
            else:
                temp ^= Pauli('I')
        H += -coupling*temp.to_matrix()
    for i in range(qubits):
        temp = Pauli('')
        for j in range(qubits):
            if (j == i):
                temp ^= Pauli('X')
            else:
                temp ^= Pauli('I')
        H += -g*temp.to_matrix()

    # make negative exponential
    g = -g
    coupling = -coupling
    
    gates = []
    if not os.path.exists("TFIM_Operators"):
        os.mkdir("TFIM_Operators")
    
    # add timestep where dt = 0
    with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
        f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
        f.write("[Trotter]\nsteps = 1\ndt = 0\n\n") # maybe need new number for steps
        f.write("[Jy]\nvalue = 0\n\n")
        f.write("[Jz]\nvalue = "+str(coupling)+"\n\n")
        f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
        f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=\nimin = 1\nimax = 2\nstep = 1\n")
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"])
    os.remove("TFIM_Operators/Operator_Generator.ini")
    qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=1.qasm")
    gate = qc.to_gate(label = "TFIM 0").control()
    gates.append(gate)
    os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=1.qasm")
    steps -= 1
    steps *= trotter
    dt /= trotter

    with open("TFIM_Operators/Operator_Generator.ini", 'w+') as f:
        f.write("[Qubits]\nnumber = "+str(qubits)+"\n\n")
        f.write("[Trotter]\nsteps = "+str(steps)+"\ndt = "+str(dt)+"\n\n") # maybe need new number for steps
        f.write("[Jy]\nvalue = 0\n\n")
        f.write("[Jz]\nvalue = "+str(coupling)+"\n\n")
        f.write("[hx]\nramp = constant\nvalue = "+str(g)+"\n\n")
        f.write("[Output]\nname = TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i=\nimin = 1\nimax = "+str(steps+1)+"\nstep = "+str(1)+"\n")
    exe = location+"/release/examples/f3c_time_evolution_TFYZ"
    subprocess.run([exe, "TFIM_Operators/Operator_Generator.ini"])
    os.remove("TFIM_Operators/Operator_Generator.ini")
    for step in range(1, steps + 1):
        if step%trotter == 0:
            qc = QuantumCircuit.from_qasm_file("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step)+".qasm")
            gate = qc.to_gate(label = "TFIM "+str(step)).control()
            gates.append(gate)
        os.remove("TFIM_Operators/n="+str(qubits)+"_g="+str(g)+"_dt="+str(dt)+"_i="+str(step)+".qasm")
    os.rmdir("TFIM_Operators")
    return gates, H

def create_hamiltonian(qubits, system, scale_factor, g=0, J=4, t=0, U=0, x=1, y=1, show_steps=False):
    assert(system[0:4].upper() == "TFIM" or system[0:4].upper() == "SPIN" or system[0:4].upper() == "HUBB")
    # assert(abs(scale_factor)<=2*pi)
    H = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    if system[0:4].upper() == "TFIM":
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
    elif system[0:4].upper() == "SPIN":
        assert(J!=0)
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
    elif system[0:4].upper() == "HUBB":
        assert(x>=0 and y>=0)
        assert(x*y == qubits)

        # coupling portion
        Sd = np.array([[0,0],[1,0]])
        S = np.array([[0,1],[0,0]])
        I = np.eye(2)
        # op1 = np.kron(Sd, S)
        # op2 = np.kron(S, Sd)
        left_right_hopping_term = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
        for op in [Sd]:
            for site in range(qubits):
                curr_x = site%x
                curr_y = site//x%y
                # couple sites in square latice
                neighbors = []
                if curr_x != 0:   neighbors.append((site-1)%qubits)
                if curr_x != x-1: neighbors.append((site+1)%qubits)
                if curr_y != 0:   neighbors.append((site+x)%qubits)
                if curr_y != y-1: neighbors.append((site-x)%qubits)

                for neighbor in neighbors:
                    temp = [1]
                    for site_ in range(qubits):
                        # print(site_)
                        # print(site_ == site, site_ == neighbor)
                        if site_ == site: temp = np.kron(temp, op)
                        elif site_ == neighbor: temp = np.kron(temp, op.T)
                        else: temp = np.kron(temp, I) 
                    if temp.shape[0] == 64: print(temp)
                    left_right_hopping_term+=temp
        left_right_hopping_term *=-t

        # number operator portion
        op1 = np.kron(Sd, Sd)
        op2 = np.kron(S, S)
        num = op1@op2
        up_down_hopping_term = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
        for place in range(qubits-1):
            temp = [1]
            for index in range(qubits-1):
                if index == place: temp = np.kron(temp, num) 
                else: temp = np.kron(temp, I) 
            up_down_hopping_term+=temp
        up_down_hopping_term*=U

        H = up_down_hopping_term+left_right_hopping_term
            
    if show_steps:
        val, vec = np.linalg.eigh(H)
        print("Original eigenvalues:", val)
        print("Original eigenvectors:\n", vec)
    
    # scale eigenvalues of the Hamiltonian
    if show_steps: print("Norm =", np.linalg.norm(H, ord=2))
    H = scale_factor*H/np.linalg.norm(H, ord=2)
    # rotate matrix so that it will be positive definite (not nessary in this usecase)
    # H += pi*np.eye(2**qubits)

    if show_steps:
        val, vec = np.linalg.eigh(H)
        print("Scaled eigenvalues:", val)
        print("Scaled eigenvectors:\n", vec)
        min_eigenvalue = np.min(val)
        print("Lowest energy eigenvalue", min_eigenvalue); print()
    
    return H