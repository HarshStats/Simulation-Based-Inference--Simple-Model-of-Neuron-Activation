import numpy as np
from tqdm import tqdm
from src.lif_model import lif_simulate

def generate_lif_dataset(n_sims=1000, T=200.0, dt=0.1, tref=2.0):
    """
    Generates a dataset of LIF neuron simulations with random parameters.

    Args:
        n_sims (int): The number of simulations to generate.
        T (float): Total simulation time (ms).
        dt (float): Time step (ms).
        tref (float): Refractory period (ms).

    Returns:
        parameters (np.ndarray): Array of shape (n_sims, 6) containing the sampled parameters.
        traces (np.ndarray): Array of shape (n_sims, n_steps) containing the voltage traces.
    """
    n_steps = int(T / dt)
    parameters = np.zeros((n_sims, 6))
    traces = np.zeros((n_sims, n_steps))

    # Define prior ranges for the parameters we want to infer
    # [tau_m, E_L, g_L, V_th, V_reset, I]
    prior_ranges = {
        'tau_m': [5.0, 30.0],
        'E_L': [-80.0, -65.0],
        'g_L': [5.0, 15.0],
        'V_th': [-60.0, -50.0],
        'V_reset': [-80.0, -65.0],
        'I': [50.0, 300.0]
    }

    print(f"Generating {n_sims} simulations...")
    for i in tqdm(range(n_sims)):
        # Sample parameters from uniform priors
        tau_m = np.random.uniform(*prior_ranges['tau_m'])
        E_L = np.random.uniform(*prior_ranges['E_L'])
        g_L = np.random.uniform(*prior_ranges['g_L'])
        V_th = np.random.uniform(*prior_ranges['V_th'])
        V_reset = np.random.uniform(*prior_ranges['V_reset'])
        I = np.random.uniform(*prior_ranges['I'])

        # Store the sampled parameters
        parameters[i, :] = [tau_m, E_L, g_L, V_th, V_reset, I]

        # Run simulation
        _, V, _ = lif_simulate(T=T, dt=dt, E_L=E_L, V_th=V_th, V_reset=V_reset, 
                                 tau_m=tau_m, g_L=g_L, I=I, tref=tref)
        
        # Store the voltage trace
        traces[i, :] = V

    return parameters, traces
