""" Main routines for QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the ground-state energy with reduced circuit depth. 

Last revision: 11/22/2022
"""
import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import erf
import fejer_kernel
import fourier_filter
import generate_cdf

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit.library import UnitaryGate, QFT
from scipy.linalg import expm

ham_shift = 3*np.pi/4

def modify_spectrum(ham):
    arr_ham = ham.toarray().astype(np.complex128)
    norm_ham = (ham_shift)*arr_ham/np.linalg.norm(arr_ham, ord = 2)
    return norm_ham

def initial_state_angle(p):
    return 2 * np.arccos((np.sqrt(2*p) + np.sqrt(2 * (1 - p)))/2)

def create_HT_circuit(qubits, unitary, W = 'Re', p0 = 1, backend = AerSimulator()):
    qr_ancilla = QuantumRegister(1)
    #qr_eigenstate = QuantumRegister(np.log2(Ham[0].shape[0]))
    qr_eigenstate = QuantumRegister(qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)
    qc.h(qr_ancilla)
    if p0 == 1:
        qc.h(qr_eigenstate)
    else:
        qc.ry(initial_state_angle(p0), qr_eigenstate)
    qc.append(unitary, qargs = [qr_ancilla[:]] + qr_eigenstate[:] )
    # if W = Imaginary
    if W[0] == 'I': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])
    print(qc)
    trans_qc = transpile(qc, backend)
    return trans_qc

def get_estimated_ground_energy_rough(d,delta,spectrum,population,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)

    compute_prob_X = lambda T: generate_cdf.compute_prob_X_(T,spectrum,population)
    compute_prob_Y = lambda T: generate_cdf.compute_prob_Y_(T,spectrum,population)


    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch) #Generate sample to calculate CDF

    total_evolution_time = np.sum(np.abs(J_arr))
    average_evolution_time = total_evolution_time/(Nsample*Nbatch)
    maxi_evolution_time=max(np.abs(J_arr[0,:]))

    Nx = 10
    Lx = np.pi/3
    ground_energy_estimate = 0.0
    count = 0
    #---"binary" search
    while Lx > delta:
        x = (2*np.arange(Nx)/Nx-1)*Lx +  ground_energy_estimate
        y_avg = generate_cdf.compute_cdf_from_XY(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs)#Calculate the value of CDF
        indicator_list = y_avg > (population[0]/2.05)
        ix = np.nonzero(indicator_list)[0][0]
        ground_energy_estimate = x[ix]
        Lx = Lx/2
        count += 1
    
    return ground_energy_estimate, count*total_evolution_time, maxi_evolution_time

def generate_filtered_Z_est(spectrum,population,t,x,d,delta,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)
    compute_prob_X = lambda t_: generate_cdf.compute_prob_X_(t_,spectrum,population)
    compute_prob_Y = lambda t_: generate_cdf.compute_prob_Y_(t_,spectrum,population)
    #Calculate <\psi|F(H)\exp(-itH)|\psi>
    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY_QCELS(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch,t) #Generate samples using Hadmard test
    y_avg = generate_cdf.compute_cdf_from_XY_QCELS(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs) 
    total_time = np.sum(np.abs(J_arr))+t*Nsample*Nbatch
    max_time= max(np.abs(J_arr[0,:]))+t
    return y_avg, total_time, max_time


def generate_Z_theory(spectrum,population,t,Nsample):
    Re=0
    Im=0
    z=np.dot(population,np.exp(-1j*spectrum*t))
    Re_true=(1+z.real)/2
    Im_true=(1+z.imag)/2
    #Simulate Hadmard test
    for nt in range(Nsample):
        if np.random.uniform(0,1)<Re_true:
           Re+=1
    for nt2 in range(Nsample):
        if np.random.uniform(0,1)<Im_true:
           Im+=1
    Z_est = complex(2*Re/Nsample-1,2*Im/Nsample-1)
    max_time = t
    total_time = t * Nsample
    return Z_est, total_time, max_time 

def generate_spectrum_population(eigenenergies, population, p):
    
    p = np.array(p)
    spectrum = eigenenergies * (ham_shift) / np.max(np.abs(eigenenergies))#normalize the spectrum
    q = population
    num_p = p.shape[0]
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    Z_fit=np.zeros(NT,dtype = 'complex') # 'complex_'
    Z_fit=(x[0]+1j*x[1])*np.exp(-1j*x[2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt(ts, Z_est, x0, bounds = None, method = 'SLSQP'):

    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)

    return res

def get_tau(j, time_steps, iterations, T):
    return T/time_steps/(2**(iterations-j))

def qcels_largeoverlap_new(Z_est, time_steps, lambda_prior, T):
    """Multi-level QCELS for a system with a large initial overlap.

    Description: The code of using Multi-level QCELS to estimate the ground state energy for a systems with a large initial overlap

    Args: eigenvalues of the Hamiltonian: spectrum; 
    overlaps between the initial state and eigenvectors: population; 
    the depth for generating the data set: T; 
    number of data pairs(time steps): time_steps; 
    number of samples: Nsample; 
    initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0; 
    maximal evolution time T_{max}; 
    total evolution time T_{total}

    """
    t_ns = time_steps
    iterations = len(Z_est) - 1
    tau = get_tau(0, time_steps, iterations, T)
    ts=tau*np.arange(time_steps)
    print("      Preprocessing", flush = True)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est[0], x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 
    for iter in range(1, iterations + 1):
        print('      Starting Iteration', "("+str(iter)+'/'+str(iterations)+")", flush = True)
        tau = get_tau(iter, time_steps, iterations, T)
        ts=tau*np.arange(time_steps)
        t_ns += time_steps
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
        res = qcels_opt(ts, Z_est[iter], x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 
    print("      Finished Iterations", flush = True)
    return res, t_ns

def qcels_smalloverlap(spectrum, population, T, NT, d, rel_gap, err_tol_rough, Nsample_rough, Nsample):
    """Multi-level QCELS with a filtered data set for a system with a small initial overlap.

    Description: The codes of using Multi-level QCELS and eigenvalue filter to estimate the ground state energy for
    a system with a small initial overlap

    Args: eigenvalues of the Hamiltonian: spectrum; 
    overlaps between the initial state and eigenvectors: population; 
    the depth for generating the data set: T; 
    number of data pairs: NT; 
    number of samples for constructing the eigenvalue filter: Nsample_rough; 
    number of samples for generating the data set: Nsample; 
    initial guess of \lambda_0: lambda_prior
    
    Returns: an estimation of \lambda_0; 
    maximal evolution time T_{max}; 
    total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.

    lambda_prior, total_time_prior, max_time_prior = get_estimated_ground_energy_rough(
            d,err_tol_rough,spectrum,population,Nsample_rough,Nbatch=1) #Get the rough estimation of the ground state energy
    x = lambda_prior + rel_gap/2 #center of the eigenvalue filter
    total_time_all += total_time_prior
    max_time_all = max(max_time_all, max_time_prior)
    
    N_level=int(np.log2(T/NT))
    Z_est=np.zeros(NT,dtype = 'complex')
    tau=T/NT/(2**N_level)
    ts=tau*np.arange(NT)
    for i in range(NT):
        Z_est[i], total_time, max_time=generate_filtered_Z_est(
                spectrum,population,ts[i],x,d,err_tol_rough,Nsample_rough,Nbatch=1)#Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est, x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau)
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau)
    #Iteration
    for n_QCELS in range(N_level):
        Z_est=np.zeros(NT,dtype = 'complex')
        tau=T/NT/(2**(N_level-n_QCELS-1))
        ts=tau*np.arange(NT)
        for i in range(NT):
            Z_est[i], total_time, max_time=generate_filtered_Z_est(
                    spectrum,population,ts[i],x,d,err_tol_rough,Nsample,Nbatch=1)#Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
            total_time_all += total_time
            max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max))
        res = qcels_opt(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau)
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau)

    return ground_energy_estimate_QCELS, total_time_all, max_time_all


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import tfim_1d
    from qcels import *
    import qiskit
    from qiskit_ibm_runtime import QiskitRuntimeService as QRS
    import matplotlib
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10

    num_sites = 4
    J = 1.0
    g = 4

    # calculate the ground state with g = 1
    ham0 = tfim_1d.generate_ham(num_sites, J, 1.0)
    ground_state_0 = ham0.eigh(subset_by_index = (0,0))[1][:,0] # g = 1 ground state


    # plot original spectrum
    ham = tfim_1d.generate_ham(num_sites, J, g)
    eigenenergies, eigenstates = ham.eigh()
    ground_state = eigenstates[:,0]
    population_raw = np.abs(np.dot(eigenstates.conj().T, ground_state_0))**2

    old_ham = ham

    # create modified spectrum
    ham = modify_spectrum(old_ham)
    eigenenergies, eigenstates = eigh(ham)

    ground_state = eigenstates[:,0]
    population = np.abs(np.dot(eigenstates.conj().T, ground_state_0))**2
