from qiskit import QuantumCircuit
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