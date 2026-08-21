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

# Step 24 - mixed_precision_step
def mixed_precision_step(x, y, master_params, scale, lr):
    """
    Runs a single mixed precision training step: half-precision forward/backward 
    with loss scaling, gradient unscaling, overflow checking, and full-precision SGD update.
    
    Args:
        x: Input batch (NumPy array).
        y: Target batch (NumPy array).
        master_params: Dictionary of float32 master parameter arrays.
        scale: Loss scaling factor.
        lr: Learning rate for SGD.
        
    Returns:
        A tuple of (unscaled_loss, new_master_params, skipped_flag).
    """
    # 1. Create half-precision views of master parameters and inputs
    params_fp16 = cast_to_half_precision(master_params)
    x_fp16 = x.astype(np.float16)
    y_fp16 = y.astype(np.float16)
    
    # 2. Run forward pass in fp16
    y_pred, cache = mlp_forward(x_fp16, params_fp16)
    
    # 3. Compute loss and upstream gradient
    loss_val, dy_pred = mse_loss_and_grad(y_pred, y_fp16)
    
    # 4. Scale loss and upstream gradient to prevent underflow
    scaled_loss, scaled_dy = scale_loss(loss_val, dy_pred, scale)
    
    # 5. Run backward pass in fp16
    grads_fp16 = mlp_backward(scaled_dy, cache, params_fp16)
    
    # 6. Unscale gradients back to float32
    grads_fp32 = unscale_gradients(grads_fp16, scale)
    
    # 7. Check for overflow (NaN or Inf)
    skipped = has_non_finite_gradients(grads_fp32)
    
    # 8. Update master parameters (ensuring dtype is explicitly float32) or skip update on overflow
    new_master = {}
    for k, param in master_params.items():
        if skipped:
            new_master[k] = param.astype(np.float32)
        else:
            new_master[k] = (param - lr * grads_fp32[k]).astype(np.float32)
            
    return float(loss_val), new_master, skipped

# Step 25 - shard_dataset_across_workers
def shard_dataset_across_workers(x, y, num_workers):
    """
    Splits the dataset (x, y) along the batch axis into num_workers contiguous shards.
    Distributes any remainder evenly among the first workers so that earlier workers 
    receive one extra sample and every sample belongs to exactly one worker.

    Args:
        x: Input dataset array of shape (N, ...)
        y: Target dataset array of shape (N, ...)
        num_workers: Integer number of workers

    Returns:
        A list of length num_workers containing tuples of (x_shard, y_shard).
    """
    n_samples = x.shape[0]
    base_size = n_samples // num_workers
    remainder = n_samples % num_workers

    shards = []
    start = 0
    for i in range(num_workers):
        # Distribute the remainder to the first 'remainder' workers
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        
        x_shard = x[start:end]
        y_shard = y[start:end]
        shards.append((x_shard, y_shard))
        
        start = end

    return shards

# Step 26 - compute_local_gradients
def compute_local_gradients(x, y, params):
    """Compute parameter gradients for one worker's data shard.

    Forward (mlp_forward) -> loss gradient (mse_loss_and_grad) -> backward
    (mlp_backward). Return a grads dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    y_pred, cache = mlp_forward(x, params)
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    grads = mlp_backward(dy_pred, cache, params)
    return grads

# Step 27 - all_reduce_mean
def all_reduce_mean(per_worker_grads):
    """
    Performs a logical all-reduce that averages a list of gradient dictionaries 
    produced by different data-parallel workers elementwise.

    Args:
        per_worker_grads: A list of dictionaries, where each dictionary maps parameter 
                          names to NumPy arrays of gradients.

    Returns:
        A single dictionary with the same keys, where each value is the elementwise 
        mean across all workers.
    """
    if not per_worker_grads:
        return {}
    
    keys = per_worker_grads[0].keys()
    
    # Average elementwise across workers for each parameter key
    return {
        key: np.mean([worker_dict[key] for worker_dict in per_worker_grads], axis=0)
        for key in keys
    }

# Step 28 - ring_all_reduce_mean
def ring_all_reduce_mean(per_worker_arrays):
    """
    Averages a list of identically shaped arrays across simulated workers using 
    a ring reduce-scatter followed by ring all-gather over equal chunks.

    Args:
        per_worker_arrays: List of NumPy arrays of the same shape.

    Returns:
        A single NumPy array of the same shape containing the elementwise mean.
    """
    if not per_worker_arrays:
        return np.array([])
    
    original_shape = per_worker_arrays[0].shape
    num_workers = len(per_worker_arrays)
    
    # 1. Flatten each worker's array
    flat_arrays = [arr.flatten() for arr in per_worker_arrays]
    
    if num_workers == 1:
        return per_worker_arrays[0].copy()

    # 2. Split each worker's array into num_workers chunks using np.array_split
    chunks = [list(np.array_split(arr, num_workers)) for arr in flat_arrays]
    
    # 3. Reduce-Scatter Phase
    for s in range(num_workers - 1):
        next_chunks = [list(w_chunks) for w_chunks in chunks]
        for i in range(num_workers):
            send_chunk_idx = (i - s) % num_workers
            recv_chunk_idx = (i - s - 1) % num_workers
            src_worker = (i - 1) % num_workers
            
            incoming = chunks[src_worker][recv_chunk_idx]
            next_chunks[i][recv_chunk_idx] = chunks[i][recv_chunk_idx] + incoming
        chunks = next_chunks

    # 4. All-Gather Phase
    for s in range(num_workers - 1):
        next_chunks = [list(w_chunks) for w_chunks in chunks]
        for i in range(num_workers):
            send_chunk_idx = (i - s + 1) % num_workers
            recv_chunk_idx = (i - s) % num_workers
            src_worker = (i - 1) % num_workers
            
            incoming = chunks[src_worker][recv_chunk_idx]
            next_chunks[i][recv_chunk_idx] = incoming
        chunks = next_chunks

    # 5. Reconstruct the final averaged array from worker 0
    final_flat = np.concatenate(chunks[0])
    final_mean = final_flat / num_workers
    
    return final_mean.reshape(original_shape)

# Step 29 - data_parallel_train_step
def data_parallel_train_step(x, y, params, num_workers, lr):
    """
    Performs one synchronous data parallel SGD update.
    
    Args:
        x: Full input batch dataset array.
        y: Full target batch dataset array.
        params: Dictionary of MLP parameters ('W1', 'b1', 'W2', 'b2').
        num_workers: Integer number of data-parallel workers.
        lr: Learning rate for SGD.
        
    Returns:
        A new parameter dictionary updated via all-reduced gradients.
    """
    # 1. Shard the dataset across the specified number of workers
    shards = shard_dataset_across_workers(x, y, num_workers)
    
    # 2. Compute local parameter gradients for each worker's shard
    per_worker_grads = [compute_local_gradients(x_shard, y_shard, params) for x_shard, y_shard in shards]
    
    # 3. Synchronize local gradients across all workers using all-reduce mean
    avg_grads = all_reduce_mean(per_worker_grads)
    
    # 4. Update parameters using a vanilla SGD step
    new_params = {}
    for k in params:
        new_params[k] = params[k] - lr * avg_grads[k]
        
    return new_params

# Step 30 - bucket_gradients
def bucket_gradients(grads, bucket_size):
    """
    Packs flattened gradient arrays from grads into fixed-size 1D buckets.
    
    Args:
        grads: Dictionary mapping parameter names to NumPy arrays.
        bucket_size: Maximum number of elements per bucket.
        
    Returns:
        A tuple of (buckets, meta) where buckets is a list of 1D NumPy arrays 
        and meta is a list of (name, shape, start, end, bucket_index) tuples.
    """
    if not grads:
        return [], []
    
    buckets = []
    meta = []
    
    current_bucket_list = []
    current_size = 0
    current_bucket_index = 0
    
    for name in sorted(grads.keys()):
        arr = grads[name]
        shape = arr.shape
        flat = arr.flatten()
        size = flat.size
        
        if current_size > 0 and current_size + size > bucket_size:
            buckets.append(np.concatenate(current_bucket_list))
            current_bucket_list = []
            current_size = 0
            current_bucket_index += 1
            
        start = current_size
        end = start + size
        current_bucket_list.append(flat)
        current_size = end
        
        meta.append((name, shape, start, end, current_bucket_index))
        
    if current_bucket_list:
        buckets.append(np.concatenate(current_bucket_list))
        
    return buckets, meta

# Step 31 - init_adam_state
def init_adam_state(params):
    """
    Builds the Adam optimizer state for a given parameter dictionary.
    
    Args:
        params: Dictionary mapping parameter names to NumPy arrays.
        
    Returns:
        A dictionary containing:
            - 'm': Dictionary of first moment vectors initialized to zeros.
            - 'v': Dictionary of second moment vectors initialized to zeros.
            - 't': Integer step counter initialized to 0.
    """
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0
    return {'m': m, 'v': v, 't': t}

# Step 32 - partition_optimizer_state
def partition_optimizer_state(state, num_workers):
    """
    Partitions Adam's first and second moment tensors across num_workers so each 
    worker owns one contiguous slice of the flattened optimizer state for every parameter,
    along with their original shapes and shard slices.
    
    Args:
        state: Dictionary containing 'm' (first moments), 'v' (second moments), and 't' (step counter).
        num_workers: Integer number of workers to partition across.
        
    Returns:
        A list of length num_workers where each element is a dictionary containing 
        'm', 'v', 'shard_slices', 'shapes', and 't'.
    """
    m_state = state['m']
    v_state = state['v']
    t = state['t']
    
    # Capture the original shapes of each parameter array
    shapes = {k: arr.shape for k, arr in m_state.items()}
    
    # Initialize worker state dictionaries
    workers = [{'m': {}, 'v': {}, 'shard_slices': {}, 'shapes': shapes, 't': t} for _ in range(num_workers)]
    
    for k in m_state.keys():
        m_flat = m_state[k].flatten()
        v_flat = v_state[k].flatten()
        n = m_flat.size
        
        base_size = n // num_workers
        remainder = n % num_workers
        
        start = 0
        for i in range(num_workers):
            size = base_size + (1 if i < remainder else 0)
            end = start + size
            
            workers[i]['m'][k] = m_flat[start:end]
            workers[i]['v'][k] = v_flat[start:end]
            workers[i]['shard_slices'][k] = (start, end)
            
            start = end
            
    return workers

# Step 33 - local_shard_adam_update
def local_shard_adam_update(params, grads, worker_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Performs one Adam optimizer update on only the local shard of each parameter 
    owned by this worker.
    
    Args:
        params: Dictionary of full parameter NumPy arrays.
        grads: Dictionary of full gradient NumPy arrays.
        worker_state: Dictionary containing 'm', 'v', 't', 'shard_slices', and 'shapes'.
        lr: Learning rate.
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        eps: Small constant for numerical stability.
        
    Returns:
        A tuple of (updated_param_shards, updated_worker_state).
    """
    t = worker_state['t'] + 1
    
    new_m = {}
    new_v = {}
    updated_param_shards = {}
    
    for k, param in params.items():
        start, end = worker_state['shard_slices'][k]
        
        # Extract parameter shard
        p_flat = param.flatten()
        p_shard = p_flat[start:end]
        
        # Extract gradient shard
        g_flat = grads[k].flatten()
        g_shard = g_flat[start:end]
        
        # Get existing moment shards
        m_shard = worker_state['m'][k]
        v_shard = worker_state['v'][k]
        
        # Update biased first and second moment estimates
        m_new = beta1 * m_shard + (1 - beta1) * g_shard
        v_new = beta2 * v_shard + (1 - beta2) * (g_shard ** 2)
        
        new_m[k] = m_new
        new_v[k] = v_new
        
        # Compute bias-corrected estimates
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        
        # Update parameter shard
        p_shard_updated = p_shard - lr * m_hat / (np.sqrt(v_hat) + eps)
        updated_param_shards[k] = p_shard_updated
        
    updated_worker_state = worker_state.copy()
    updated_worker_state['m'] = new_m
    updated_worker_state['v'] = new_v
    updated_worker_state['t'] = t
    
    return updated_param_shards, updated_worker_state

# Step 34 - all_gather_param_shards
def all_gather_param_shards(param_shards_per_worker, shapes, shard_slices_per_worker):
    # Initialize the result dictionary
    gathered_params = {}
    
    # Get the parameter names from the first worker's shards
    param_names = list(param_shards_per_worker[0].keys())
    
    for param_name in param_names:
        # Get the original shape for this parameter
        original_shape = shapes[param_name]
        
        # Calculate the total flattened length
        total_length = int(np.prod(original_shape))
        
        # Preallocate a flat buffer for this parameter
        flat_buffer = np.zeros(total_length, dtype=np.float64)
        
        # Iterate over workers to place their shards
        for worker_idx, shards in enumerate(param_shards_per_worker):
            # Get the slice boundaries for this parameter and worker
            start, end = shard_slices_per_worker[worker_idx][param_name]
            
            # Get the shard for this parameter from this worker
            shard = shards[param_name]
            
            # Place the shard at the correct position in the flat buffer
            flat_buffer[start:end] = shard
        
        # Reshape the flat buffer to the original shape
        gathered_params[param_name] = flat_buffer.reshape(original_shape)
    
    return gathered_params

# Step 35 - zero_optimizer_step
def zero_optimizer_step(params, grads, worker_states, lr=0.001,
                         beta1=0.9, beta2=0.999, eps=1e-8):
    shapes = {name: arr.shape for name, arr in params.items()}

    param_shards_per_worker = []
    shard_slices_per_worker = []
    new_worker_states = []

    for ws in worker_states:
        # Hand the FULL params/grads + this worker's own state to the helper.
        # It owns the slicing (via ws['shard_slices']) and the Adam math.
        updated_shards, updated_ws = local_shard_adam_update(
            params, grads, ws, lr, beta1, beta2, eps
        )

        param_shards_per_worker.append(updated_shards)
        shard_slices_per_worker.append(ws['shard_slices'])
        new_worker_states.append(updated_ws)

    new_params = all_gather_param_shards(
        param_shards_per_worker, shapes, shard_slices_per_worker
    )

    return new_params, new_worker_states

# Step 36 - compute_param_memory_bytes
def compute_param_memory_bytes(params):
    # TODO: sum the total bytes occupied by every parameter array in the dict.
    total_bytes = 0
    for arr in params.values():
        total_bytes += arr.nbytes
    return total_bytes

# Step 37 - compute_optimizer_memory_bytes
def compute_optimizer_memory_bytes(state, num_workers=1, sharded=False):
    # TODO: return per-worker bytes of Adam state (m and v), dividing by num_workers if sharded.
    total_bytes = 0
    
    # Sum bytes for 'm' moment tensors
    for arr in state['m'].values():
        total_bytes += arr.nbytes
    
    # Sum bytes for 'v' moment tensors
    for arr in state['v'].values():
        total_bytes += arr.nbytes
    
    # If sharded, divide by number of workers (each worker stores a fraction)
    if sharded:
        total_bytes //= num_workers
    
    return total_bytes

# Step 38 - compute_peak_activation_memory_bytes
def compute_peak_activation_memory_bytes(x, params, checkpointed=False):
    # TODO: return total bytes of activations retained by the forward cache
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    if checkpointed:
        # Checkpointed forward: only retain input x
        cache = {'x': x}
        # We still need to compute the forward pass but only keep x
        # The rest is not stored
        z1 = x @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        # z2 not stored in cache
    else:
        # Standard forward: retain x, z1, a1, z2
        z1 = x @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}
    
    total_bytes = 0
    for arr in cache.values():
        total_bytes += arr.nbytes
    return total_bytes

# Step 39 - compare_memory_with_and_without_optimizations (not yet solved)
# TODO: implement

# Step 40 - full_distributed_training_loop (not yet solved)
# TODO: implement

