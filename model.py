"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_synthetic_regression_batch
def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    # Seed numpy for reproducibility
    np.random.seed(seed)
    
    # Sample inputs from standard normal distribution
    x = np.random.randn(batch_size, in_dim).astype(np.float64)
    
    # Build a hidden linear teacher mapping from in_dim to out_dim
    W_true = np.random.randn(in_dim, out_dim).astype(np.float64)
    
    # Generate targets from teacher plus Gaussian noise
    noise_scale = 0.01  # small amount of noise
    y = x @ W_true + noise_scale * np.random.randn(batch_size, out_dim).astype(np.float64)
    
    return x, y

# Step 2 - init_mlp_params
def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    # Seed numpy's global RNG for reproducibility
    np.random.seed(seed)
    
    # He initialization for W1: std = sqrt(2 / fan_in) where fan_in = in_dim
    std_w1 = np.sqrt(2.0 / in_dim)
    W1 = np.random.randn(in_dim, hidden_dim).astype(np.float64) * std_w1
    
    # Bias for first layer starts at zero
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    
    # He initialization for W2: std = sqrt(2 / fan_in) where fan_in = hidden_dim
    std_w2 = np.sqrt(2.0 / hidden_dim)
    W2 = np.random.randn(hidden_dim, out_dim).astype(np.float64) * std_w2
    
    # Bias for second layer starts at zero
    b2 = np.zeros(out_dim, dtype=np.float64)
    
    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

# Step 3 - linear_forward
def linear_forward(x, w, b):
    # TODO: apply y = x @ w + b and return the resulting (N, out_dim) array
    return x@w+b

# Step 4 - relu_forward
def relu_forward(x):
    # TODO: apply the ReLU activation elementwise and return an array of the same shape.
    return np.maximum(0,x)

# Step 5 - mlp_forward
def mlp_forward(x, params):
    # Extract parameters
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    # First linear layer: z1 = x @ W1 + b1
    z1 = linear_forward(x, W1, b1)
    
    # ReLU activation: a1 = ReLU(z1)
    a1 = relu_forward(z1)
    
    # Second linear layer: z2 = a1 @ W2 + b2 (final prediction)
    z2 = linear_forward(a1, W2, b2)
    
    # Cache intermediate tensors needed for backward pass
    cache = {
        'x': x,      # input to first layer
        'z1': z1,    # pre-activation of first layer (needed for ReLU gradient)
        'a1': a1,    # post-activation of first layer (input to second layer)
        'z2': z2     # final prediction
    }
    
    return z2, cache

# Step 6 - mse_loss_and_grad
def mse_loss_and_grad(y_pred, y_true):
    # Compute the residual
    residual = y_pred - y_true
    
    # Compute loss: average of squared residuals over all elements
    loss = np.mean(residual ** 2).item()  # convert to Python float
    
    # Compute gradient: dL/dy_pred = 2 * residual / (N * D)
    # np.mean divides by the total number of elements, so we use the same normalization
    n_elements = residual.size
    dy_pred = 2 * residual / n_elements
    
    return loss, dy_pred

# Step 7 - linear_backward
import numpy as np

def linear_backward(d_out, x, w):
    # d_out shape: (N, out_dim)
    # x shape: (N, in_dim)
    # w shape: (in_dim, out_dim)
    
    # Gradient with respect to input: dx = d_out @ w.T
    # Shape: (N, out_dim) @ (out_dim, in_dim) -> (N, in_dim)
    dx = d_out @ w.T
    
    # Gradient with respect to weight: dw = x.T @ d_out
    # Shape: (in_dim, N) @ (N, out_dim) -> (in_dim, out_dim)
    dw = x.T @ d_out
    
    # Gradient with respect to bias: db = sum of d_out over batch dimension
    # Shape: (out_dim,)
    db = np.sum(d_out, axis=0)
    
    return dx, dw, db

# Step 8 - relu_backward
def relu_backward(d_out, z):
    # ReLU derivative: 1 where z > 0, 0 elsewhere
    # Gradient: dz = d_out * (z > 0)
    dz = d_out * (z > 0)
    return dz

# Step 9 - first_linear_backward
def first_linear_backward(d_z1, x, w1):
    # Shape: (N, hidden_dim) @ (hidden_dim, in_dim) -> (N, in_dim)
    dx = d_z1 @ w1.T
    
    # Gradient with respect to weight: dW1 = x.T @ d_z1
    # Shape: (in_dim, N) @ (N, hidden_dim) -> (in_dim, hidden_dim)
    dW1 = x.T @ d_z1
    
    # Gradient with respect to bias: db1 = sum of d_z1 over batch dimension
    # Shape: (hidden_dim,)
    db1 = np.sum(d_z1, axis=0)
    
    return dx, dW1, db1

# Step 10 - mlp_backward
def mlp_backward(dy_pred, cache, params):
    # Extract cached values
    x = cache['x']        # input to first layer, shape (N, in_dim)
    z1 = cache['z1']      # pre-activation of first layer, shape (N, hidden_dim)
    a1 = cache['a1']      # post-activation of first layer, shape (N, hidden_dim)
    z2 = cache['z2']      # final prediction, shape (N, out_dim)
    
    # Extract parameters
    W1 = params['W1']     # shape (in_dim, hidden_dim)
    W2 = params['W2']     # shape (hidden_dim, out_dim)
    
    # Backward through second linear layer: z2 = a1 @ W2 + b2
    # Input to this layer: a1 (post-ReLU), weight: W2
    da1, dW2, db2 = linear_backward(dy_pred, a1, W2)
    # da1 shape: (N, hidden_dim)
    # dW2 shape: (hidden_dim, out_dim)
    # db2 shape: (out_dim,)
    
    # Backward through ReLU: a1 = ReLU(z1)
    # Need pre-activation z1 from cache
    dz1 = relu_backward(da1, z1)
    # dz1 shape: (N, hidden_dim)
    
    # Backward through first linear layer: z1 = x @ W1 + b1
    # Use helper for first layer specifically
    dx, dW1, db1 = first_linear_backward(dz1, x, W1)
    # dx shape: (N, in_dim) - not needed for parameter updates
    # dW1 shape: (in_dim, hidden_dim)
    # db1 shape: (hidden_dim,)
    
    # Return gradients dictionary matching params structure
    grads = {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2
    }
    
    return grads

# Step 11 - split_into_micro_batches
def split_into_micro_batches(x, y, micro_batch_size):
    # Get total number of samples
    n_samples = x.shape[0]
    
    # Initialize list to store micro batches
    micro_batches = []
    
    # Iterate over the full batch in steps of micro_batch_size
    for start in range(0, n_samples, micro_batch_size):
        end = min(start + micro_batch_size, n_samples)
        x_mb = x[start:end]
        y_mb = y[start:end]
        micro_batches.append((x_mb, y_mb))
    
    return micro_batches

# Step 12 - accumulate_gradients
def accumulate_gradients(accum_grads, new_grads):
    # If no accumulator exists yet, start with the new gradients (copy to avoid mutation)
    if accum_grads is None:
        return {key: value.copy() for key, value in new_grads.items()}
    
    # Otherwise, sum elementwise
    accumulated = {}
    for key in accum_grads.keys():
        # Accumulate gradients by elementwise addition
        accumulated[key] = accum_grads[key] + new_grads[key]
    
    return accumulated

# Step 13 - scale_accumulated_gradients
def scale_accumulated_gradients(accum_grads, num_micro_batches):
    # Divide each gradient tensor by the number of micro batches
    scaled_grads = {}
    for key, grad in accum_grads.items():
        scaled_grads[key] = grad / num_micro_batches
    
    return scaled_grads

# Step 14 - grad_accumulation_step
def grad_accumulation_step(x, y, params, micro_batch_size):
    micro_batches = split_into_micro_batches(x, y, micro_batch_size)
    K = len(micro_batches)
    total_grads = None

    for x_mb, y_mb in micro_batches:
        y_pred, cache = mlp_forward(x_mb, params)
        loss, dy_pred = mse_loss_and_grad(y_pred, y_mb)
        new_grads = mlp_backward(dy_pred, cache, params)

        # accumulate the RAW micro-batch gradient (no scaling yet)
        total_grads = accumulate_gradients(total_grads, new_grads)

    # apply the averaging exactly once, at the end
    total_grads = {k: v / K for k, v in total_grads.items()}
    return total_grads

# Step 15 - mlp_forward_checkpointed
def mlp_forward_checkpointed(x, params):
    # Extract parameters
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    # First linear layer: z1 = x @ W1 + b1
    z1 = linear_forward(x, W1, b1)
    
    # ReLU activation: a1 = ReLU(z1)
    a1 = relu_forward(z1)
    
    # Second linear layer: z2 = a1 @ W2 + b2 (final prediction)
    z2 = linear_forward(a1, W2, b2)
    
    # Cache only the block input x (minimal for recomputation)
    cache = {'x': x}
    
    return z2, cache

# Step 16 - recompute_block_activations
def recompute_block_activations(x, params):
    # Extract parameters
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    # Recompute first linear layer: z1 = x @ W1 + b1
    z1 = linear_forward(x, W1, b1)
    
    # Recompute ReLU activation: a1 = ReLU(z1)
    a1 = relu_forward(z1)
    
    # Recompute second linear layer: z2 = a1 @ W2 + b2
    z2 = linear_forward(a1, W2, b2)
    
    # Return full cache with all intermediate tensors
    cache = {
        'x': x,      # block input
        'z1': z1,    # pre-activation of first layer
        'a1': a1,    # post-activation of first layer (hidden)
        'z2': z2     # final prediction
    }
    
    return cache

# Step 17 - mlp_backward_checkpointed
def recompute_block_activations(x, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    # 1. First linear layer
    z1 = linear_forward(x, W1, b1)
    
    # 2. ReLU activation
    a1 = relu_forward(z1)
    
    # 3. Second linear layer (must take post-activation a1)
    z2 = linear_forward(a1, W2, b2)
    
    return {
        'x': x,
        'z1': z1,
        'a1': a1,
        'z2': z2
    }

def mlp_backward_checkpointed(dy_pred, light_cache, params):
    # Extract saved input tensor
    x = light_cache['x']
    
    # Recompute full activation cache
    full_cache = recompute_block_activations(x, params)
    
    # Pass recomputed cache and dy_pred to standard backward pass
    return mlp_backward(dy_pred, full_cache, params)

# Step 18 - estimate_checkpointing_memory_savings
def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    """
    Estimates activation memory in bytes retained during the forward pass of the 
    two-layer MLP for both full caching and checkpointed caching.

    Args:
        batch_size: Number of samples in the batch (N)
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension
        dtype_bytes: Number of bytes per element (e.g., 4 for float32)

    Returns:
        A dictionary with integer keys 'full_bytes', 'checkpoint_bytes', and 'saved_bytes'.
    """
    # Full forward caches block input (x), hidden pre-activation (z1), and post-activation (a1)
    full_elements = (batch_size * in_dim) + (batch_size * hidden_dim) + (batch_size * hidden_dim)
    full_bytes = full_elements * dtype_bytes
    
    # Checkpointed forward caches only the block input (x)
    checkpoint_elements = batch_size * in_dim
    checkpoint_bytes = checkpoint_elements * dtype_bytes
    
    saved_bytes = full_bytes - checkpoint_bytes
    
    return {
        'full_bytes': int(full_bytes),
        'checkpoint_bytes': int(checkpoint_bytes),
        'saved_bytes': int(saved_bytes)
    }

# Step 19 - cast_to_half_precision
def cast_to_half_precision(values):
    """
    Converts a dictionary of NumPy arrays into a new dictionary where every array 
    is converted to float16, preserving all keys and shapes.
    
    Args:
        values: Dictionary mapping string keys to NumPy arrays.
        
    Returns:
        A new dictionary with the same keys and arrays cast to np.float16.
    """
    return {k: v.astype(np.float16) for k, v in values.items()}

# Step 20 - make_master_params
def make_master_params(params):
    """
    Builds the float32 master copy of every parameter tensor for a mixed precision trainer.
    
    Args:
        params: Dictionary mapping string keys to NumPy arrays (potentially float16).
        
    Returns:
        A new dictionary with the same keys and shapes, containing independent copies 
        of each array stored in np.float32.
    """
    return {k: v.astype(np.float32) for k, v in params.items()}

# Step 21 - scale_loss
def scale_loss(loss, dy_pred, scale):
    """
    Multiplies both the scalar loss and the upstream gradient dy_pred by a fixed loss scale factor.
    
    Args:
        loss: Scalar loss value.
        dy_pred: NumPy array of upstream gradients.
        scale: Loss scaling factor (scalar).
        
    Returns:
        A tuple of (scaled_loss, scaled_dy_pred).
    """
    scaled_loss = loss * scale
    scaled_dy_pred = dy_pred * scale
    return scaled_loss, scaled_dy_pred

# Step 22 - unscale_gradients
def unscale_gradients(grads, scale):
    """
    Divides every gradient tensor in the given dictionary by the loss scale factor 
    and returns a new dictionary with independent arrays stored in np.float32.
    
    Args:
        grads: Dictionary mapping string keys to gradient NumPy arrays.
        scale: Loss scaling factor (scalar).
        
    Returns:
        A new dictionary with the same keys, where each gradient array is divided 
        by scale and cast to float32.
    """
    return {k: (v / scale).astype(np.float32) for k, v in grads.items()}

# Step 23 - has_non_finite_gradients
def has_non_finite_gradients(grads):
    """
    Scans a dictionary of gradient arrays and returns True if any gradient contains
    a NaN or Inf value, otherwise False.
    
    Args:
        grads: Dictionary mapping string keys to gradient NumPy arrays.
        
    Returns:
        bool: True if any non-finite value is found, False otherwise.
    """
    for v in grads.values():
        if not np.all(np.isfinite(v)):
            return True
    return False

# Step 24 - mixed_precision_step (not yet solved)
# TODO: implement

# Step 25 - shard_dataset_across_workers (not yet solved)
# TODO: implement

# Step 26 - compute_local_gradients (not yet solved)
# TODO: implement

# Step 27 - all_reduce_mean (not yet solved)
# TODO: implement

# Step 28 - ring_all_reduce_mean (not yet solved)
# TODO: implement

# Step 29 - data_parallel_train_step (not yet solved)
# TODO: implement

# Step 30 - bucket_gradients (not yet solved)
# TODO: implement

# Step 31 - init_adam_state (not yet solved)
# TODO: implement

# Step 32 - partition_optimizer_state (not yet solved)
# TODO: implement

# Step 33 - local_shard_adam_update (not yet solved)
# TODO: implement

# Step 34 - all_gather_param_shards (not yet solved)
# TODO: implement

# Step 35 - zero_optimizer_step (not yet solved)
# TODO: implement

# Step 36 - compute_param_memory_bytes (not yet solved)
# TODO: implement

# Step 37 - compute_optimizer_memory_bytes (not yet solved)
# TODO: implement

# Step 38 - compute_peak_activation_memory_bytes (not yet solved)
# TODO: implement

# Step 39 - compare_memory_with_and_without_optimizations (not yet solved)
# TODO: implement

# Step 40 - full_distributed_training_loop (not yet solved)
# TODO: implement

