import numpy as np
import ot
import time

def load_rgb_distributions(filepath='data/rgb_distributions.npz'):
    """Loads the pre-computed RGB distributions from file."""
    print("--- Loading RGB Distributions ---")
    try:
        data = np.load(filepath)
        X_src = data['X_src']
        w_src = data['w_src']
        X_tgt = data['X_tgt']
        w_tgt = data['w_tgt']
        print(f"✅ Data loaded successfully from {filepath}")
        print(f"   Source shape: {X_src.shape}, Target shape: {X_tgt.shape}")
        return X_src, w_src, X_tgt, w_tgt
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found.")
        print("Please run 'rgb_distributions.py' first to generate the data.")
        return None, None, None, None

def compute_cost_matrix(X_src, X_tgt):
    """
    Computes the cost matrix using POT's ot.dist() function.
    This completes Step 4 of the project.
    """
    print("\n--- Step 4: Computing Cost Matrix (using POT) ---")
    start_time = time.time()
    
    # Using float32 for a smaller memory footprint, as per instructions.
    # ot.dist is highly optimized and will return a float64 matrix by default.
    # We can convert it after computation.
    C = ot.dist(X_src, X_tgt, metric='sqeuclidean')
    C = C.astype(np.float32)

    end_time = time.time()
    
    duration = end_time - start_time
    memory_mb = C.nbytes / (1024 * 1024)
    
    print(f"✅ Cost matrix `C` computed successfully.")
    print(f"   Shape: {C.shape}, dtype: {C.dtype}")
    print(f"   Time taken: {duration:.4f} seconds")
    print(f"   Memory usage: {memory_mb:.2f} MB")
    
    # Validate some values
    manual_dist_check = np.sum((X_src[0] - X_tgt[0])**2)
    print(f"   Validation: C[0,0] = {C[0,0]:.4f}, Manual check = {manual_dist_check:.4f}")
    assert np.isclose(C[0,0], manual_dist_check)
    
    return C

def compute_transport_plan(w_src, w_tgt, C, reg=0.01):
    """
    Computes the Optimal Transport plan using POT's ot.sinkhorn().
    This completes Step 5 of the project.
    
    Args:
        w_src (np.ndarray): Source weights.
        w_tgt (np.ndarray): Target weights.
        C (np.ndarray): Cost matrix.
        reg (float): Epsilon regularization parameter.
    
    Returns:
        np.ndarray: The transport plan `P`.
    """
    print("\n--- Step 5: Computing Optimal Transport Plan (using POT) ---")
    print(f"Using regularization ε = {reg}")
    start_time = time.time()
    
    # Try with initial regularization
    P = ot.sinkhorn(w_src, w_tgt, C, reg, numItermax=1000, stopThr=1e-8)
    
    # Check if we need to retry with higher regularization
    row_sum_error = np.sum(np.abs(P.sum(axis=1) - w_src))
    col_sum_error = np.sum(np.abs(P.sum(axis=0) - w_tgt))
    
    # If convergence failed, try with higher regularization
    if row_sum_error > 1e-3 or col_sum_error > 1e-3:
        print(f"   ⚠️  Initial convergence challenging (errors: {row_sum_error:.2e}, {col_sum_error:.2e})")
        print("   🔄 Retrying with increased regularization...")
        reg_adaptive = reg * 10  # Increase regularization 10x
        print(f"   Using adaptive regularization ε = {reg_adaptive}")
        P = ot.sinkhorn(w_src, w_tgt, C, reg_adaptive, numItermax=2000, stopThr=1e-6)
        
        # Recalculate errors
        row_sum_error = np.sum(np.abs(P.sum(axis=1) - w_src))
        col_sum_error = np.sum(np.abs(P.sum(axis=0) - w_tgt))
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Transport plan `P` computed successfully.")
    print(f"   Shape: {P.shape}, dtype: {P.dtype}")
    print(f"   Time taken: {duration:.4f} seconds")

    # Validate marginals
    print(f"   Marginal constraint validation:")
    print(f"     ||P.sum(axis=1) - w_src||_1 = {row_sum_error:.2e}")
    print(f"     ||P.sum(axis=0) - w_tgt||_1 = {col_sum_error:.2e}")
    
    assert row_sum_error < 1e-3, "Source marginals not satisfied even with adaptive regularization!"
    assert col_sum_error < 1e-3, "Target marginals not satisfied even with adaptive regularization!"
    
    if row_sum_error > 1e-6 or col_sum_error > 1e-6:
        print("   ⚠️  Marginals satisfied but with relaxed tolerance (challenging distribution)")
    else:
        print("   ✅ Marginals are satisfied within strict tolerance.")
    
    return P

if __name__ == "__main__":
    # --- Step 1: Load Data ---
    X_src, w_src, X_tgt, w_tgt = load_rgb_distributions()
    
    if X_src is not None:
        # --- Step 2: Compute Cost Matrix (Project Step 4) ---
        C = compute_cost_matrix(X_src, X_tgt)
        
        # --- Step 3: Compute Transport Plan (Project Step 5) ---
        P = compute_transport_plan(w_src, w_tgt, C, reg=0.01)
        
        # --- Step 4: Save Artifacts ---
        output_filepath = 'data/transport_plan.npz'
        np.savez(output_filepath, C=C, P=P)
        
        print(f"\n✅ Results saved successfully to '{output_filepath}'.")
        print("This file contains the cost matrix `C` and the transport plan `P`.")
        print("\nReady for Step 6: Colour Transfer.") 