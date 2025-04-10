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

def flatten(xss):
    return [x for xs in xss for x in xs]

def modify_spectrum(ham):
    arr_ham = ham.toarray().astype(np.complex128)
    norm_ham = (ham_shift)*arr_ham/np.linalg.norm(arr_ham, ord = 2)
    return norm_ham

def initial_state_angle(p):
    return 2 * np.arccos((np.sqrt(2*p) + np.sqrt(2 * (1 - p)))/2)

def create_HT_circuit(qubits, unitary, W = 'Re', backend = AerSimulator(), init_state = []):
    qr_ancilla = QuantumRegister(1)
    qr_eigenstate = QuantumRegister(qubits)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr_ancilla, qr_eigenstate, cr)
    qc.h(qr_ancilla)
    qc.initialize(init_state, qr_eigenstate[:])
    #qc.h(qr_eigenstate)
    qc.append(unitary, qargs = [qr_ancilla[:]] + qr_eigenstate[:])
    # if W = Imaginary
    if W[0] == 'I': qc.sdg(qr_ancilla)
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla[0],cr[0])
    #print(qc)
    trans_qc = transpile(qc, backend, optimization_level=3)
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
    from Ham_generator import *
    from qcels import *
    import qiskit
    from qiskit_ibm_runtime import QiskitRuntimeService as QRS
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.circuit.library import Initialize
    import matplotlib
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10

    num_sites = 4

    #TFIM paramters
    J_T = 1.0
    g_T = 4

    #HSM paramters
    J_H = 4
    g_H = 0

    # T (TFIM), H (HSM)
    model_type = 'T'

    if model_type[0].upper() == 'T':
        mn = 'TFIM'
        print('Transverse Field Ising Model')

        # calculate the ground state with g = 1
        ham0 = tfim_1d.generate_ham(num_sites, J_T, 1.0)
        ground_state_0 = ham0.eigh(subset_by_index = (0,0))[1][:,0] # g = 1 ground state

        # plot original spectrum
        ham = tfim_1d.generate_ham(num_sites, J_T, g_T)
        eigenenergies, eigenstates = ham.eigh()
        ground_state = eigenstates[:,0]
        population_raw = np.abs(np.dot(eigenstates.conj().T, ground_state_0))**2

        old_ham = ham

        # create modified spectrum
        ham = modify_spectrum(old_ham)
        eigenenergies, eigenstates = eigh(ham)

        ground_state = eigenstates[:,0]
        population = np.abs(np.dot(eigenstates.conj().T, ground_state_0))**2

    if model_type[0].upper() == 'H':
        mn = 'HSM'
        print('Heisenberg Spin Model')

        ham = create_hamiltonian(num_sites, 'SPIN', g = g_H, J=J_H, show_steps=False)
        spin_energies, spin_states = eigh(ham)
        ground_state = spin_states[:,0]

        pop = np.abs(np.dot(spin_states.conj().T, ground_state))**2
    
    computation_type = 'S'
    output_file = True
    p0_array            = np.array([0.1, 0.5, 0.9]) # initial overlap with the first eigenvector
    # p0_array            = np.arange(0.6, 0.99, 0.05)
    deltas              = 1 - np.sqrt(p0_array)
    trials              = 8 # number of comparisions each test (circuit depths)
    tests               = 1
    err_threshold       = 0.01
    T0                  = 100

    # QCELS variables
    time_steps          = 5
    epsilons            = np.array([0.1, 0.02, 0.009, 0.006, 0.003, 0.001, 0.0001, 0.00004])
    iterations          = [int(np.log2(1/time_steps/i)) for i in epsilons]
    err_QCELS           = np.zeros((len(p0_array),trials))
    est_QCELS           = np.zeros((len(p0_array),trials))
    cost_list_avg_QCELS = np.zeros((len(p0_array),trials))
    rate_success_QCELS  = np.zeros((len(p0_array),trials))
    max_T_QCELS         = np.zeros((len(p0_array),trials))

    # initialization: S (Quantum Simulation), or R (Quantum Hardware)

    if computation_type[0].upper() == 'S':
        print("\nQUANTUM SIMULATION SELECTED\n")

        backend = AerSimulator()
        data_name = "Q_Sim"

    if computation_type[0].upper() == 'R':
        print("\nQUANTUM HARDWARE SELECTED\n")    
        
        # save qiskit API token for later use
        api_token = input("Enter API Token:")
        service = QRS(channel = 'ibm_quantum', instance='rpi-rensselaer/research/faulsf', token = api_token)
        backend = service.backend('ibm_rensselaer')
        data_name = "Q_Real"

    if output_file:
        outfile = open("Output/"+str(data_name)+"_"+str(mn)+"_trans.txt", 'w')

    ansatz = []
    for p in range(len(p0_array)):
        psi = ground_state

        # Generate a random vector orthogonal to psi
        random_vec = np.random.randn(2**num_sites) + 1j * np.random.randn(2**num_sites)
        random_vec -= np.vdot(psi, random_vec) * psi  # Make orthogonal to psi
        random_vec /= np.linalg.norm(random_vec)  # Normalize

        # Construct psi with the required squared overlap
        overlap_squared = p0_array[p]
        phi = np.sqrt(overlap_squared) * psi + np.sqrt(1 - overlap_squared) * random_vec

        ansatz.append(phi)

    # Transpiles circuits
    for p in range(len(p0_array)):
        p0=p0_array[p]
        n_success_QCELS= np.zeros(trials)
        n_success_QPE= np.zeros(trials)

        print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")")

        if output_file: print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")", file = outfile)

        print("  Generating QCELS circuits", "(p0="+str(p0)+")")

        #spectrum, population = generate_spectrum_population(eigenenergies, population_raw, [p0])

        #------------------QCELS-----------------
        Nsample = 100 # number of samples for constructing the loss function

        for trial in range(trials):
            print("    Transpiling QCELS", "("+str(trial+1)+"/"+str(trials)+")")
            epsilon = epsilons[trial]

            if output_file: print("    Transpiling QCELS", "("+str(trial+1)+"/"+str(trials)+")", file = outfile, flush = True)
            T = 1/epsilon
            for j in range(iterations[trial] + 1):
                tau = get_tau(j, time_steps, iterations[trial], T)
                qcs_QCELS = []
                #unitaries = (generate_TFIM_gates(num_sites, time_steps, tau, g_T, '../../../f3cpp'))
                for data_pair in range(time_steps):
                    t = tau*data_pair
                    mat = expm(-1j*ham*t)
                    controlled_U = UnitaryGate(mat).control(annotated="yes")
                    #qcs_QCELS.append(create_HT_circuit(num_sites, unitaries[data_pair], W = 'Re', backend = backend, init_state = ground_state))
                    #qcs_QCELS.append(create_HT_circuit(num_sites, unitaries[data_pair], W = 'Im', backend = backend, init_state = ground_state))
                    qcs_QCELS.append(create_HT_circuit(num_sites, controlled_U, W = 'Re', backend = backend, init_state = ansatz[p]))
                    qcs_QCELS.append(create_HT_circuit(num_sites, controlled_U, W = 'Im', backend = backend, init_state = ansatz[p]))
                with open('Transpiled_Circuits/QCELS_p0='+str(p0)+'_Trial'+str(trial)+'_Iter='+str(j)+'.qpy', 'wb') as f:
                    qiskit.qpy.dump(qcs_QCELS, f)
        print('Finished transpiling for QCELS ', "(p0="+str(p0)+")")
    
    # Loads transpiled circuits
    QCELS_depths = []
    qcs_QCELS = []

    for p in range(len(p0_array)):
        p0 = p0_array[p]
        QCELS_depths.append([])
        for test in range(tests):
            QCELS_depths[p].append([])
            for trial in range(trials):
                depth = 0
                print('Loading QCELS data ('+str(trial+1)+'/'+str(trials)+')')
                for i in range(iterations[trial] + 1):
                    with open('Transpiled_Circuits/QCELS_p0='+str(p0)+'_Trial'+str(trial)+'_Iter='+str(i)+'.qpy', 'rb') as f:
                        circs = qiskit.qpy.load(f)
                        for time_step in range(time_steps):
                            depth += circs[time_step].depth() 
                        qcs_QCELS.append(circs)

                QCELS_depths[p][test].append(depth)

    qcs_QCELS = sum(qcs_QCELS, []) # flatten list

    num_splits = len(p0_array)*tests
    split = int(len(qcs_QCELS)/num_splits)

    qcs_QCELS_circuits = []
    for i in range(num_splits):
        qcs_QCELS_circuits.append(qcs_QCELS[i*split:(i+1)*split])

    # Runs loaded circuits
    print('Running transpiled circuits')
    sampler = Sampler(backend)
    jobs = []
    results = []
    for i in range(num_splits):
        job = sampler.run(qcs_QCELS_circuits[i], shots = T0)
        result = job.result()
        jobs.append(job)
        results.append(result)
    results = flatten(results)

    Z_ests = []
    QCELS_times = []

    for p in range(len(p0_array)):
        Z_ests.append([])
        QCELS_times.append([])
        for test in range(tests):
            Z_ests[p].append([])
            QCELS_times[p].append([])
            for trial in range(trials):
                exec_time = 0
                Z_ests[p][test].append([])
                for iter in range(iterations[trial] + 1):
                    Z_ests[p][test][trial].append([])
                    for time_step in range(time_steps):
                        index = time_step*2 + iter*time_steps*2 + (sum(iterations[0:trial])+trial)*time_steps*2 + test*(sum(iterations)+len(iterations))*time_steps*2 + p*tests*(sum(iterations)+len(iterations))*time_steps*2
                        raw_data_re = results[index].data
                        counts_re = raw_data_re[list(raw_data_re.keys())[0]].get_counts()
                        raw_data_im = results[index + 1].data
                        counts_im = raw_data_im[list(raw_data_im.keys())[0]].get_counts()
                        exec_time += (0 + 1)

                        re_p0 = im_p0 = 0
                        if counts_re.get('0') is not None:
                            re_p0 = counts_re['0']/T0
                        if counts_im.get('0') is not None:
                            im_p0 = counts_im['0']/T0
                        
                        Re = 2*re_p0-1
                        Im = 2*im_p0-1 

                        Angle = np.arccos(Re)
                        if  np.arcsin(Im)<0:
                            Phase = 2*np.pi - Angle
                        else:
                            Phase = Angle

                        Z_est = complex(np.cos(Phase),np.sin(Phase))
                        Z_ests[p][test][trial][iter].append(Z_est)
                QCELS_times[p][test].append(exec_time)

    lambda_prior = -ham_shift

    if output_file:
        outfile = open("Output/"+str(data_name)+"_"+str(mn)+"_run.txt", 'w')

    for p in range(len(p0_array)):
        p0=p0_array[p]
        n_success_QCELS= np.zeros(trials)
        n_success_QPE= np.zeros(trials)

        print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")")

        if output_file: print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")", file = outfile)

        for test in range(tests):

            print("  Generating QCELS and QPE data", "(p0="+str(p0)+")","("+str(test+1)+"/"+str(tests)+")")

            #spectrum, population = generate_spectrum_population(eigenenergies, population_raw, [p0])

            #------------------QCELS-----------------
            Nsample = 100 # number of samples for constructing the loss function

            for trial in range(trials):
                print("    Running QCELS", "("+str(trial+1)+"/"+str(trials)+")")

                if output_file: print("    Running QCELS", "("+str(trial+1)+"/"+str(trials)+")", file = outfile, flush = True)
                epsilon = epsilons[trial]
                T = 1/epsilon
                #lambda_prior = spectrum[0]
                ground_energy_estimate_QCELS, cosT_depth_list_this = qcels_largeoverlap_new(Z_ests[p][test][trial], time_steps, lambda_prior, T)
                #ground_energy_estimate_QCELS, cosT_depth_list_this, max_T_QCELS_this = qcels_largeoverlap()
                
                #cosT_depth_list_this = QCELS_depths[p][test][trial]
                max_T_QCELS_this = 0

                print("      Estimated ground state energy =", ground_energy_estimate_QCELS)
                if output_file: print("      Estimated ground state energy =", ground_energy_estimate_QCELS.x[2], file = outfile)

                est_this_run_QCELS = ground_energy_estimate_QCELS.x[2]
                err_this_run_QCELS = np.abs(ground_energy_estimate_QCELS.x[2] - lambda_prior)
                err_QCELS[p,trial] = err_QCELS[p,trial]+np.abs(err_this_run_QCELS)
                est_QCELS[p,trial] = est_QCELS[p,trial] + est_this_run_QCELS
                cost_list_avg_QCELS[p,trial]=cost_list_avg_QCELS[p,trial]+cosT_depth_list_this
                max_T_QCELS[p,trial]=max(max_T_QCELS[p,trial],max_T_QCELS_this)

                if np.abs(err_this_run_QCELS)<err_threshold:
                    n_success_QCELS[trial]+=1

            print("    Finished QCELS data\n")
            if output_file: print("    Finished QCELS data\n", file = outfile)

        rate_success_QCELS[p,:] = n_success_QCELS[:]/tests
        err_QCELS[p,:] = err_QCELS[p,:]/tests
        est_QCELS[p,:] = est_QCELS[p,:]/tests
        #cost_list_avg_QCELS[p,:]=cost_list_avg_QCELS[p,:]/tests
        cost_list_avg_QCELS[p,:]=2*cost_list_avg_QCELS[p,:]/tests # observables instead of time steps

    if model_type[0].upper() == 'T':
        np.savez('Data/'+data_name+'_result_TFIM_'+str(num_sites)+'sites_QCELS',name1=rate_success_QCELS,name2=max_T_QCELS,name3=cost_list_avg_QCELS,name4=err_QCELS,name5=est_QCELS)
        #np.savez('Data/'+data_name+'_TFIM_8sites_data',name1=spectrum,name2=population,name3=ground_energy_estimate_QCELS.x[0],
                #name4=ground_energy_estimate_QCELS.x[1],name5=ground_energy_estimate_QCELS.x[2])
    if model_type[0].upper() == 'H':
        np.savez('Data/'+data_name+'_result_HSM_'+str(num_sites)+'sites_QCELS',name1=rate_success_QCELS,name2=max_T_QCELS,name3=cost_list_avg_QCELS,name4=err_QCELS,name5=est_QCELS)

    print("Saved data to files starting with", data_name)
    if output_file: print("Saved data to files starting with", data_name, file = outfile, flush=True)
    outfile.close()
    if output_file: print("Saved output to file ", "Output/"+str(data_name)+".txt")





