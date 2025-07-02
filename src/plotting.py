import matplotlib.pyplot as plt
import numpy as np

def plot_lif_simulation(t, V, spikes, V_th):
    """
    Plots the membrane potential trace of a LIF neuron simulation.

    Args:
        t (np.ndarray): Time array
        V (np.ndarray): Membrane potential trace
        spikes (np.ndarray): Spike times (ms)
        V_th (float): Spike threshold (mV)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, V, label='Membrane Potential')
    ax.axhline(V_th, color='r', linestyle='--', label=f'Threshold ({V_th} mV)')
    if len(spikes) > 0:
        ax.plot(spikes, np.ones_like(spikes) * V_th, 'ro', label='Spikes')
    ax.set_title('Leaky Integrate-and-Fire (LIF) Neuron Simulation')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.legend()
    ax.grid(True)
    plt.show()
