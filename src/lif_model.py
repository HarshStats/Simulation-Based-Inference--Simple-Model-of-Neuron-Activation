import numpy as np

def lif_simulate(T=100.0, dt=0.1, E_L=-75.0, V_th=-55.0, V_reset=-75.0, tau_m=10.0, g_L=10.0, I=100.0, tref=2.0):
    """
    Simulate a Leaky Integrate-and-Fire (LIF) neuron.

    Args:
        T (float): Total simulation time (ms)
        dt (float): Time step (ms)
        E_L (float): Resting potential (leak reversal potential) (mV)
        V_th (float): Spike threshold (mV)
        V_reset (float): Reset potential after spike (mV)
        tau_m (float): Membrane time constant (ms)
        g_L (float): Leak conductance (nS)
        I (float or np.ndarray): Input current (pA or array)
        tref (float): Refractory period (ms)

    Returns:
        t (np.ndarray): Time array
        V (np.ndarray): Membrane potential trace
        spikes (np.ndarray): Spike times (ms)
    """
    n_steps = int(T / dt)
    t = np.arange(0, T, dt)
    V = np.zeros(n_steps)
    V[0] = E_L
    spikes = []
    last_spike = -np.inf
    if isinstance(I, (float, int)):
        I = I * np.ones(n_steps)
    for i in range(1, n_steps):
        if (t[i] - last_spike) < tref:
            V[i] = V_reset
            continue
        dV = (-(V[i-1] - E_L) + I[i-1] / g_L) / tau_m * dt
        V[i] = V[i-1] + dV
        if V[i] >= V_th:
            V[i] = V_reset
            spikes.append(t[i])
            last_spike = t[i]
    return t, V, np.array(spikes)


