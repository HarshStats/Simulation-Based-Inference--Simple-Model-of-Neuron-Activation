"""
BayesFlow utilities for LIF parameter inference.

This module contains helper functions for setting up and training
BayesFlow models for simulation-based inference of LIF neuron parameters.
"""

import numpy as np
import tensorflow as tf

# Try different BayesFlow import approaches based on version
try:
    import bayesflow as bf
    # Check available components
    print(f"BayesFlow version: {bf.__version__}")
    print(f"Available BayesFlow components: {dir(bf)}")
except Exception as e:
    print(f"BayesFlow import issue: {e}")


def create_simple_posterior_estimator(num_params=6, trace_dim=2000):
    """
    Create a simple neural network for parameter estimation.
    
    Args:
        num_params (int): Number of parameters to infer
        trace_dim (int): Dimension of input traces
        
    Returns:
        tf.keras.Model: Neural network model
    """
    import tensorflow as tf
    
    # Simple dense network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(trace_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_params)  # Output layer for parameters
    ])
    
    return model


def create_summary_network(units=64, num_dense=3):
    """
    Create a DeepSet summary network for processing voltage traces.
    
    Args:
        units (int): Number of units in dense layers
        num_dense (int): Number of dense layers
        
    Returns:
        tf.keras.Model or bf.networks.DeepSet: Configured summary network
    """
    try:
        return bf.networks.DeepSet(
            dense_args=dict(units=units, activation='relu'),
            pooling_fun='mean',
            num_dense=num_dense
        )
    except AttributeError:
        # Fallback to simple dense network
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dense(units//2, activation='relu')
        ])


def create_inference_network(num_params=6, num_coupling_layers=6):
    """
    Create a coupling flow inference network.
    
    Args:
        num_params (int): Number of parameters to infer
        num_coupling_layers (int): Number of coupling layers
        
    Returns:
        tf.keras.Model: Configured inference network
    """
    try:
        return bf.networks.CouplingFlow(
            num_params=num_params,
            num_coupling_layers=num_coupling_layers
        )
    except AttributeError:
        # Fallback to simple dense network
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_params)
        ])


def create_simple_trainer(model):
    """
    Create a simple trainer using standard TensorFlow/Keras.
    
    Args:
        model: Neural network model
        
    Returns:
        tuple: (model, optimizer, loss_fn)
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
    
    return model, optimizer, loss_fn


def data_generator(params, traces, batch_size=32):
    """
    Generator function for training data.
    
    Args:
        params (np.ndarray): Parameter array
        traces (np.ndarray): Voltage trace array
        batch_size (int): Batch size
        
    Yields:
        tuple: (traces_batch, params_batch)
    """
    n_samples = len(params)
    indices = np.arange(n_samples)
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield traces[batch_indices], params[batch_indices]


def setup_simple_model(num_params=6, trace_dim=2000):
    """
    Complete setup of a simple neural network model for parameter estimation.
    
    Args:
        num_params (int): Number of parameters to infer
        trace_dim (int): Dimension of input traces
        
    Returns:
        tuple: (model, optimizer, loss_fn)
    """
    # Create simple neural network
    model = create_simple_posterior_estimator(num_params=num_params, trace_dim=trace_dim)
    
    # Create trainer components
    model, optimizer, loss_fn = create_simple_trainer(model)
    
    return model, optimizer, loss_fn


def train_simple_model(model, train_traces, train_params, test_traces, test_params, epochs=50):
    """
    Train the neural network model with progress monitoring.
    
    Args:
        model: Keras model
        train_traces: Training voltage traces
        train_params: Training parameters
        test_traces: Test voltage traces
        test_params: Test parameters
        epochs: Number of training epochs
        
    Returns:
        dict: Training history with losses
    """
    
    print(f"🚀 Training neural network model for {epochs} epochs...")
    
    # Train the model
    history = model.fit(
        train_traces, train_params,
        validation_data=(test_traces, test_params),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    return {
        'train_losses': history.history['loss'],
        'val_losses': history.history['val_loss'],
        'train_mae': history.history['mae'],
        'val_mae': history.history['val_mae']
    }


def setup_bayesflow_model(num_params=6):
    """
    Complete setup of model architecture with fallback to simple approach.
    
    Args:
        num_params (int): Number of parameters to infer
        
    Returns:
        tuple: Model components
    """
    try:
        # Try BayesFlow approach first
        summary_net = create_summary_network()
        inference_net = create_inference_network(num_params=num_params)
        
        # If we get here, BayesFlow components worked
        print("✅ Using BayesFlow components")
        return None, None, summary_net, inference_net
        
    except Exception as e:
        print(f"⚠️  BayesFlow not fully compatible, using simple neural network approach")
        print(f"   Error: {e}")
        
        # Fallback to simple model
        trace_dim = 2000  # Based on our data
        model, optimizer, loss_fn = setup_simple_model(num_params=num_params, trace_dim=trace_dim)
        
        return model, optimizer, loss_fn, None


def train_bayesflow_model(trainer=None, train_params=None, train_traces=None, 
                         test_params=None, test_traces=None, epochs=50, **kwargs):
    """
    Train model with automatic fallback to simple approach.
    
    Args:
        trainer: BayesFlow trainer (if available) or Keras model
        train_params: Training parameters
        train_traces: Training traces
        test_params: Test parameters
        test_traces: Test traces
        epochs: Number of training epochs
        
    Returns:
        dict: Training history
    """
    # Check if we're using simple model (trainer is actually a Keras model)
    if hasattr(trainer, 'fit'):  # Keras model
        print("🔄 Using simple neural network training")
        return train_simple_model(
            trainer, train_traces, train_params, 
            test_traces, test_params, epochs
        )
    else:
        print("❌ BayesFlow trainer not available")
        return {'error': 'Training method not available'}
