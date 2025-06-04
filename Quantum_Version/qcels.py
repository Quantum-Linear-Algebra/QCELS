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

def get_q_job(job_id, service):
    print("Loading data from job")
    job = service.job(job_id)
    return job.result()

def create_HT_circuit(qubits, unitary, W = 'Re', backend = AerSimulator(), init_state = []):
    """
    Description: The code to create a Hadamard test circuits for a unitary operator 

    Args: number of qubits to represent the eigenstate: qubits; 
    time evolution unitary operator: unitary; 
    specifies real (imaginary) HT: W = 'Re'('Im'); 
    pecifies simulation (hardware) backend: backend = AerSimulator() (ibm_'hardware');
    eigenstate initialization with p0 overlap with ground_state: init_state

    Returns: a transpiled HT circuit: trans_qc
    """
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

def get_tau(j, time_steps, epsilon, delta):
    return delta*(2**(j - 1 - np.ceil(np.log2(1/epsilon))))/(time_steps*(epsilon))

def qcels_largeoverlap(Z_est, time_steps, lambda_prior, epsilon, delta):
    """Multi-level QCELS for a system with a large initial overlap.

    Description: The code of using Multi-level QCELS to estimate the ground state energy for a systems with a large initial overlap

    Args: expectation values of time evolution: Z_est; 
    1/precision: T; 
    number of data pairs(time steps): time_steps; 
    initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0: res; 
    total time steps performed: t_ns; 
    """
    t_ns = time_steps
    iterations = len(Z_est) - 1
    tau = get_tau(0, time_steps, epsilon, delta)
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
        tau = get_tau(iter, time_steps, epsilon, delta)
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

def base_qcels_largeoverlap(Z_est, time_steps, lambda_prior, epsilon, delta):
    """Multi-level QCELS for a system with a large initial overlap.

    Description: The code of using Multi-level QCELS to estimate the ground state energy for a systems with a large initial overlap

    Args: expectation values of time evolution: Z_est; 
    1/precision: T; 
    number of data pairs(time steps): time_steps; 
    initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0: res; 
    total time steps performed: t_ns; 
    """
    t_ns = time_steps
    tau = get_tau(0, time_steps, epsilon, delta)
    ts=tau*np.arange(time_steps)
    print("      Preprocessing", flush = True)
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
    print('      Starting Optimization', flush = True)
    tau = get_tau(1, time_steps, epsilon, delta)
    ts=tau*np.arange(time_steps)
    t_ns += time_steps
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
    print("      Finished Optimization", flush = True)
    return res, t_ns


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from Ham_generator import *
    from qcels import *
    import qiskit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import ParityMapper
    from qiskit_nature.units import DistanceUnit
    from qiskit_ibm_runtime import QiskitRuntimeService as QRS
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_aer.noise import NoiseModel, depolarizing_error, QuantumError, coherent_unitary_error, ReadoutError, thermal_relaxation_error
    import matplotlib
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10
    np.set_printoptions(threshold=np.inf)

    num_sites = 2

    #TFIM parameters
    J_T = 1
    g_T = 4

    #HSM parameters
    J_H = 4
    g_H = 0

    #Hubb parameters
    t_H = 1
    U_H = 10

    #H2 molecule parameters
    distance = 0.5

    # T (TFIM), H (HSM), B (Hubbard), M (H2 molecule)
    model_type = 'T'
    # Q (Qiskit), F(F3C++)
    Ham_type = 'F'

    if model_type[0].upper() == 'T':
        mn = 'TFIM'
        print('Transverse Field Ising Model')

        if Ham_type[0].upper() == 'F':
            unitaries, ham = (generate_TFIM_gates(num_sites, 2, 1, g_T, ham_shift, '../../../f3cpp', trotter = 1000))
            eigenenergies, eigenenstates = eigh(ham)
            ground_state = eigenenstates[:,0]
            
        if Ham_type[0].upper() == 'Q':
            ham = create_hamiltonian(num_sites, 'TFIM', ham_shift, g = g_T, J=J_T, show_steps=False)
            eigenenergies, eigenstates = eigh(ham)
            ground_state = eigenstates[:,0]

            pop = np.abs(np.dot(eigenstates.conj().T, ground_state))**2
        
    if model_type[0].upper() == 'H':
        mn = 'HSM'
        print('Heisenberg Spin Model')

        ham = create_hamiltonian(num_sites, 'SPIN', ham_shift, g = g_H, J=J_H, show_steps=False)
        eigenenergies, eigenstates = eigh(ham)
        ground_state = eigenstates[:,0]

        pop = np.abs(np.dot(eigenstates.conj().T, ground_state))**2

    if model_type[0].upper() == 'B':
        mn = "HUBB"
        print('Hubbard Model')

        ham = create_hamiltonian(num_sites, 'HUBB', ham_shift, t = t_H, U=U_H, x = num_sites, y = 1, show_steps=False)
        eigenenergies, eigenstates = eigh(ham)
        ground_state = eigenstates[:,0]

        popp = np.abs(np.dot(eigenstates.conj().T, ground_state))**2

    if model_type[0].upper() == 'M':
        mn = 'HH'
        num_sites = 1
        ang = 0.52917721092
        print('H2 Molecule')

        driver = PySCFDriver(
            atom=f'H .0 .0 .0; H .0 .0 {distance}',
            unit=DistanceUnit.ANGSTROM,
            basis='sto3g'
        )

        molecule = driver.run()
        mapper = ParityMapper(num_particles=molecule.num_particles)
        hamiltonian = molecule.hamiltonian.second_q_op()
        tapered_mapper = molecule.get_tapered_mapper(mapper)
        operator = tapered_mapper.map(hamiltonian)
        ham = operator.to_matrix()
        
        eigenenergies, eigenstates = eigh(ham)
        ground_state = eigenstates[:,0]

        popp = np.abs(np.dot(eigenstates.conj().T, ground_state))**2
    
    # initialization: S (Quantum Simulation), or R (Quantum Hardware)
    computation_type = 'R'
    output_file = True
    p0_array            = np.array([0.75, 0.9]) # initial overlap with the first eigenvector
    deltas              = np.sqrt(1-p0_array)
    trials              = 5 # number of comparisions each test (circuit depths)
    tests               = 5
    err_threshold       = 0.01
    T0                  = 100

    # QCELS variables
    time_steps          = 5
    epsilons            = np.logspace(-1,-3, trials)
    print('Chosen epsilons', epsilons)
    iterations          = [int(np.ceil(np.log2(1/i)) + 1) for i in epsilons]
    epsilons            = [2**(1 - i) for i in iterations]
    err_QCELS           = np.zeros((len(p0_array),trials))
    est_QCELS           = np.zeros((len(p0_array),trials))
    cost_list_avg_QCELS = np.zeros((len(p0_array),trials))
    rate_success_QCELS  = np.zeros((len(p0_array),trials))
    max_T_QCELS         = np.zeros((len(p0_array),trials))

    print('Iterations', iterations)
    print('Real epsilons', epsilons)

    # initialization: S (Quantum Simulation), or R (Quantum Hardware)

    if computation_type[0].upper() == 'S':
        print("\nQUANTUM SIMULATION SELECTED\n")

        noise_model = NoiseModel()
        backend = AerSimulator(noise_model=noise_model)
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

        print(np.abs(np.vdot(psi, phi))**2)
        ansatz.append(phi)

    # Create and run HT for lambda_prior

    circs = []
    if Ham_type[0].upper() == 'F':
        print('F3C++')
        for p in range(len(p0_array)):
            trans_qc1 = create_HT_circuit(num_sites, unitaries[-1], W = 'Re', backend = backend, init_state = ansatz[p])
            trans_qc2 = create_HT_circuit(num_sites, unitaries[-1], W = 'Im', backend = backend, init_state = ansatz[p])
            

            circs.append(trans_qc1)
            circs.append(trans_qc2)

        sampler = Sampler(backend)
        job = sampler.run(circs, shots = 10000)
        lambda_results = job.result()

    if Ham_type[0].upper() == 'Q':
        print('Qiskit')
        for p in range(len(p0_array)):
            mat = expm(-1j*ham)
            controlled_U = UnitaryGate(mat).control(annotated="yes")

            trans_qc1 = create_HT_circuit(num_sites, controlled_U, W = 'Re', backend = backend, init_state = ansatz[p])
            trans_qc2 = create_HT_circuit(num_sites, controlled_U, W = 'Im', backend = backend, init_state = ansatz[p])

            circs.append(trans_qc1)
            circs.append(trans_qc2)

        sampler = Sampler(backend)
        job = sampler.run(circs, shots = 10000)
        lambda_results = job.result()

    # Get lambda_prior
    lambda_priors = []

    re_data1 = lambda_results[0].data
    im_data1 = lambda_results[1].data

    re_data2 = lambda_results[2].data
    im_data2 = lambda_results[3].data


    counts_re1 = re_data1[list(re_data1.keys())[0]].get_counts()
    counts_im1 = im_data1[list(im_data1.keys())[0]].get_counts()

    counts_re2 = re_data2[list(re_data2.keys())[0]].get_counts()
    counts_im2 = im_data2[list(im_data2.keys())[0]].get_counts()


    re_p0 = im_p0 = 0
    if counts_re1.get('0') is not None:
        re_p0 = counts_re1['0']/10000
    if counts_im1.get('0') is not None:
        im_p0 = counts_im1['0']/10000

    Re = 2*re_p0 - 1
    Im = 2*im_p0 - 1

    Angle = np.arccos(Re)
    if  np.arcsin(Im)<0:
        Phase = 2*np.pi - Angle
    else:
        Phase = Angle

    lambda_prior = -Phase
    lambda_priors.append(lambda_prior)


    re_p0 = im_p0 = 0
    if counts_re2.get('0') is not None:
        re_p0 = counts_re2['0']/10000
    if counts_im2.get('0') is not None:
        im_p0 = counts_im2['0']/10000

    Re = 2*re_p0 - 1
    Im = 2*im_p0 - 1

    Angle = np.arccos(Re)
    if  np.arcsin(Im)<0:
        Phase = 2*np.pi - Angle
    else:
        Phase = Angle

    lambda_prior = -Phase
    lambda_priors.append(lambda_prior)

    print('lambda_priors: ', lambda_priors, '\n target: ', eigenenergies[0])

    # Transpiles circuits
    for p in range(len(p0_array)):
        p0=p0_array[p]
        delta = deltas[p]

        print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")")

        if output_file: print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")", file = outfile)

        print("  Generating QCELS circuits", "(p0="+str(p0)+")")

        #------------------QCELS-----------------
        for trial in range(trials):
            print("    Transpiling QCELS", "("+str(trial+1)+"/"+str(trials)+")")
            
            if output_file: print("    Transpiling QCELS", "("+str(trial+1)+"/"+str(trials)+")", file = outfile, flush = True)

            epsilon = epsilons[trial]
            for j in range(iterations[trial] + 1):
                tau = get_tau(j, time_steps, epsilon, delta)
                print('tau',tau)
                qcs_QCELS = []
                if Ham_type[0].upper() == 'F':
                    unitaries, _ = (generate_TFIM_gates(num_sites, time_steps, tau, g_T, ham_shift, '../../../f3cpp', trotter = 1000))
                for data_pair in range(time_steps):
                    if Ham_type[0].upper() == 'Q':
                        t = tau*data_pair
                        mat = expm(-1j*ham*t)
                        controlled_U = UnitaryGate(mat).control(annotated="yes")
                        qcs_QCELS.append(create_HT_circuit(num_sites, controlled_U, W = 'Re', backend = backend, init_state = ansatz[p]))
                        qcs_QCELS.append(create_HT_circuit(num_sites, controlled_U, W = 'Im', backend = backend, init_state = ansatz[p]))
                    if Ham_type[0].upper() == 'F':
                        qcs_QCELS.append(create_HT_circuit(num_sites, unitaries[data_pair], W = 'Re', backend = backend, init_state = ansatz[p]))
                        qcs_QCELS.append(create_HT_circuit(num_sites, unitaries[data_pair], W = 'Im', backend = backend, init_state = ansatz[p]))
                    
                with open('Transpiled_Circuits/QCELS_p0='+str(p0)+'_Trial'+str(trial)+'_Iter='+str(j)+'.qpy', 'wb') as f:
                    qiskit.qpy.dump(qcs_QCELS, f)
        print('Finished transpiling for QCELS ', "(p0="+str(p0)+")")
    
    # Loads transpiled circuits
    QCELS_depths = []
    qcs_QCELS = []

    for p in range(len(p0_array)):
        p0 = p0_array[p]
        print("Loading p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")")
        QCELS_depths.append([])
        for test in range(tests):
            print("  Test", test + 1)
            QCELS_depths[p].append([])
            for trial in range(trials):
                depth = 0
                print('    Loading QCELS data ('+str(trial+1)+'/'+str(trials)+')')
                for i in range(iterations[trial] + 1):
                    with open('Transpiled_Circuits/QCELS_p0='+str(p0)+'_Trial'+str(trial)+'_Iter='+str(i)+'.qpy', 'rb') as f:
                        circs = qiskit.qpy.load(f)
                        for time_step in range(time_steps):
                            depth += circs[time_step].depth() 
                        qcs_QCELS.append(circs)

                QCELS_depths[p][test].append(depth)

    qcs_QCELS = sum(qcs_QCELS, []) # flatten list

    num_splits = 1
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

    # results = list(get_q_job('d0wcfkphtw7g008py6vg', service))

    Z_ests = []
    for p in range(len(p0_array)):
        Z_ests.append([])
        for test in range(tests):
            Z_ests[p].append([])
            for trial in range(trials):
                Z_ests[p][test].append([])
                for iter in range(iterations[trial] + 1):
                    Z_ests[p][test][trial].append([])
                    for time_step in range(time_steps):
                        index = time_step*2 + iter*time_steps*2 + (sum(iterations[0:trial])+trial)*time_steps*2 + test*(sum(iterations)+len(iterations))*time_steps*2 + p*tests*(sum(iterations)+len(iterations))*time_steps*2
                        raw_data_re = results[index].data
                        counts_re = raw_data_re[list(raw_data_re.keys())[0]].get_counts()
                        raw_data_im = results[index + 1].data
                        counts_im = raw_data_im[list(raw_data_im.keys())[0]].get_counts()

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
                        Phase = Phase
                        Z_est = complex(np.cos(Phase),np.sin(Phase))
                        Z_ests[p][test][trial][iter].append(Z_est)

    if output_file:
        outfile = open("Output/"+str(data_name)+"_"+str(mn)+"_run.txt", 'w')

    for p in range(len(p0_array)):
        p0=p0_array[p]
        lambda_prior = lambda_priors[p]
        delta = deltas[p]
        
        n_success_QCELS= np.zeros(trials)

        print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")")

        if output_file: print("Testing p0 =", p0,"("+str(p+1)+"/"+str(len(p0_array))+")", file = outfile)

        for test in range(tests):

            print("  Generating QCELS data", "(p0="+str(p0)+")","("+str(test+1)+"/"+str(tests)+")")
            #------------------QCELS-----------------
            for trial in range(trials):
                print("    Running QCELS", "("+str(trial+1)+"/"+str(trials)+")")

                if output_file: print("    Running QCELS", "("+str(trial+1)+"/"+str(trials)+")", file = outfile, flush = True)
                epsilon = epsilons[trial]
                ground_energy_estimate_QCELS, cosT_depth_list_this = qcels_largeoverlap(Z_ests[p][test][trial], time_steps, lambda_prior, epsilon, delta)
                est_this_run_QCELS = ground_energy_estimate_QCELS.x[2] 

                if output_file: print("      Estimated ground state energy =", est_this_run_QCELS, file = outfile)
                
                err_this_run_QCELS = np.abs(est_this_run_QCELS - eigenenergies[0])
                err_QCELS[p,trial] = err_QCELS[p,trial]+np.abs(err_this_run_QCELS)
                est_QCELS[p,trial] = est_QCELS[p,trial] + est_this_run_QCELS
                cost_list_avg_QCELS[p,trial]=cost_list_avg_QCELS[p,trial]+cosT_depth_list_this

                if np.abs(err_this_run_QCELS)<err_threshold:
                    n_success_QCELS[trial]+=1

            print("    Finished QCELS data\n")
            if output_file: print("    Finished QCELS data\n", file = outfile)

        rate_success_QCELS[p,:] = n_success_QCELS[:]/tests
        err_QCELS[p,:] = err_QCELS[p,:]/tests
        est_QCELS[p,:] = est_QCELS[p,:]/tests
        #cost_list_avg_QCELS[p,:]=cost_list_avg_QCELS[p,:]/tests
        cost_list_avg_QCELS[p,:]=2*T0*cost_list_avg_QCELS[p,:]/tests # total shots instead of time steps (dont multiply by T0 for observables)


    if model_type[0].upper() == 'T':
        np.savez('Data/'+data_name+'_result_TFIM_'+str(num_sites)+'sites_QCELS_long',name1=rate_success_QCELS,name2=cost_list_avg_QCELS,name3=err_QCELS,name4=est_QCELS,name5=eigenenergies[0],name6=p0_array)
    if model_type[0].upper() == 'H':
        np.savez('Data/'+data_name+'_result_HSM_'+str(num_sites)+'sites_QCELS_long',name1=rate_success_QCELS,name2=cost_list_avg_QCELS,name3=err_QCELS,name4=est_QCELS,name5=eigenenergies[0],name6=p0_array)
    if model_type[0].upper() == 'B':
        np.savez('Data/'+data_name+'_result_HUBB_'+str(num_sites)+'sites_QCELS_long',name1=rate_success_QCELS,name2=cost_list_avg_QCELS,name3=err_QCELS,name4=est_QCELS,name5=eigenenergies[0],name6=p0_array)
    if model_type[0].upper() == 'M':
        np.savez('Data/'+data_name+'_result_HH_'+str(num_sites)+'sites_QCELS_long',name1=rate_success_QCELS,name2=cost_list_avg_QCELS,name3=err_QCELS,name4=est_QCELS,name5=eigenenergies[0],name6=p0_array)

    print("Saved data to files starting with", data_name)
    if output_file: print("Saved data to files starting with", data_name, file = outfile, flush=True)
    outfile.close()
    if output_file: print("Saved output to file ", "Output/"+str(data_name)+".txt")






