import numpy as np
import matplotlib.pyplot as plt
import time

def test_pot_installation():
    """
    Test POT installation and basic functionality.
    """
    print("=== Step 3.5: Setup and Explore Python Optimal Transport (POT) ===")
    
    try:
        import ot
        print(f"✅ POT successfully imported, version: {ot.__version__}")
    except ImportError as e:
        print(f"❌ POT not installed: {e}")
        print("Please install with: pip install POT")
        return False
    
    # Test basic functionality with small example
    print("\n--- Testing Basic OT Functionality ---")
    
    # Simple 2D example
    n_samples = 100
    np.random.seed(42)
    
    # Create two simple 2D distributions
    X1 = np.random.normal(0, 1, (n_samples, 2))
    X2 = np.random.normal(2, 1, (n_samples, 2))
    
    # Uniform weights
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    
    # Compute cost matrix
    C = ot.dist(X1, X2, metric='sqeuclidean')
    print(f"✅ Cost matrix computed: shape {C.shape}")
    
    # Test basic Sinkhorn
    reg = 0.1
    P = ot.sinkhorn(a, b, C, reg)
    print(f"✅ Basic Sinkhorn completed: transport plan shape {P.shape}")
    print(f"   Marginal errors: ||P1-a||₁ = {np.abs(P.sum(1) - a).sum():.2e}, ||P^T1-b||₁ = {np.abs(P.sum(0) - b).sum():.2e}")
    
    return True

def explore_sinkhorn_variants():
    """
    Explore different Sinkhorn variants available in POT.
    """
    import ot
    
    print("\n--- Exploring POT Sinkhorn Variants ---")
    
    # List available Sinkhorn functions
    sinkhorn_functions = [attr for attr in dir(ot) if 'sinkhorn' in attr.lower()]
    print(f"Available Sinkhorn functions: {sinkhorn_functions}")
    
    # Focus on log-stabilized versions
    print("\n--- Log-Stabilized Sinkhorn Functions ---")
    for func_name in sinkhorn_functions:
        if 'log' in func_name.lower() or 'stab' in func_name.lower():
            func = getattr(ot, func_name)
            print(f"✅ {func_name}: {func.__doc__.split('.')[0] if func.__doc__ else 'No description'}")

def test_sinkhorn_log_stabilized():
    """
    Test the log-stabilized Sinkhorn algorithm with our RGB data.
    """
    import ot
    
    print("\n--- Testing Log-Stabilized Sinkhorn with RGB Data ---")
    
    # Load our RGB distributions
    try:
        data = np.load('rgb_distributions.npz')
        X_src = data['X_src']
        X_tgt = data['X_tgt'] 
        w_src = data['w_src']
        w_tgt = data['w_tgt']
        print(f"✅ Loaded RGB distributions: X_src {X_src.shape}, X_tgt {X_tgt.shape}")
    except FileNotFoundError:
        print("❌ RGB distributions not found. Please run rgb_distributions.py first.")
        return None, None, None
    
    # Compute cost matrix using POT
    print("Computing cost matrix...")
    start_time = time.time()
    C = ot.dist(X_src, X_tgt, metric='sqeuclidean')
    cost_time = time.time() - start_time
    print(f"✅ Cost matrix computed in {cost_time:.2f}s: shape {C.shape}, memory ~{C.nbytes/1024/1024:.1f}MB")
    
    # Test different regularization parameters
    reg_values = [0.1, 0.05, 0.01, 0.005]
    results = {}
    
    for reg in reg_values:
        print(f"\n--- Testing ε = {reg} ---")
        
        # Standard Sinkhorn
        start_time = time.time()
        try:
            P_standard = ot.sinkhorn(w_src, w_tgt, C, reg, verbose=False)
            standard_time = time.time() - start_time
            
            # Check marginals
            err1 = np.abs(P_standard.sum(1) - w_src).sum()
            err2 = np.abs(P_standard.sum(0) - w_tgt).sum()
            
            print(f"✅ Standard Sinkhorn: {standard_time:.2f}s, marginal errors: {err1:.2e}, {err2:.2e}")
            
            results[reg] = {
                'P_standard': P_standard,
                'time_standard': standard_time,
                'error1': err1,
                'error2': err2
            }
            
        except Exception as e:
            print(f"❌ Standard Sinkhorn failed: {e}")
            
        # Log-stabilized Sinkhorn (if available)
        if hasattr(ot, 'sinkhorn_stabilized'):
            start_time = time.time()
            try:
                P_stabilized = ot.sinkhorn_stabilized(w_src, w_tgt, C, reg, verbose=False)
                stabilized_time = time.time() - start_time
                
                # Check marginals
                err1_stab = np.abs(P_stabilized.sum(1) - w_src).sum()
                err2_stab = np.abs(P_stabilized.sum(0) - w_tgt).sum()
                
                print(f"✅ Stabilized Sinkhorn: {stabilized_time:.2f}s, marginal errors: {err1_stab:.2e}, {err2_stab:.2e}")
                
                results[reg]['P_stabilized'] = P_stabilized
                results[reg]['time_stabilized'] = stabilized_time
                results[reg]['error1_stab'] = err1_stab
                results[reg]['error2_stab'] = err2_stab
                
                # Compare plans
                plan_diff = np.abs(P_standard - P_stabilized).max()
                print(f"   Max difference between plans: {plan_diff:.2e}")
                
            except Exception as e:
                print(f"❌ Stabilized Sinkhorn failed: {e}")
    
    return C, results, reg_values

def analyze_pot_api():
    """
    Analyze the POT API to understand input/output formats.
    """
    import ot
    
    print("\n--- POT API Analysis ---")
    
    # Document the key functions we'll need
    functions_to_analyze = ['sinkhorn', 'sinkhorn_stabilized', 'dist']
    
    for func_name in functions_to_analyze:
        if hasattr(ot, func_name):
            func = getattr(ot, func_name)
            print(f"\n🔍 {func_name}:")
            print(f"   Signature: {func.__name__}{func.__code__.co_varnames[:func.__code__.co_argcount]}")
            if func.__doc__:
                # Get first few lines of docstring
                doc_lines = func.__doc__.split('\n')[:5]
                for line in doc_lines:
                    if line.strip():
                        print(f"   {line.strip()}")

def create_pot_reference():
    """
    Create a reference implementation using POT for comparison.
    """
    import ot
    
    print("\n--- Creating POT Reference Implementation ---")
    
    # Load RGB data
    try:
        data = np.load('rgb_distributions.npz')
        X_src = data['X_src']
        X_tgt = data['X_tgt'] 
        w_src = data['w_src']
        w_tgt = data['w_tgt']
    except FileNotFoundError:
        print("❌ RGB distributions not found.")
        return None
    
    # Create reference implementation
    def pot_reference_ot(X_src, X_tgt, w_src, w_tgt, reg=0.01, method='stabilized'):
        """
        Reference OT implementation using POT.
        
        Args:
            X_src: Source points (N, 3)
            X_tgt: Target points (N, 3) 
            w_src: Source weights (N,)
            w_tgt: Target weights (N,)
            reg: Regularization parameter
            method: 'standard' or 'stabilized'
        
        Returns:
            P: Transport plan (N, N)
            info: Dictionary with timing and convergence info
        """
        start_time = time.time()
        
        # Compute cost matrix
        C = ot.dist(X_src, X_tgt, metric='sqeuclidean')
        cost_time = time.time() - start_time
        
        # Run Sinkhorn
        sinkhorn_start = time.time()
        if method == 'stabilized' and hasattr(ot, 'sinkhorn_stabilized'):
            P = ot.sinkhorn_stabilized(w_src, w_tgt, C, reg, verbose=False)
        else:
            P = ot.sinkhorn(w_src, w_tgt, C, reg, verbose=False)
        sinkhorn_time = time.time() - sinkhorn_start
        
        # Compute marginal errors
        err1 = np.abs(P.sum(1) - w_src).sum()
        err2 = np.abs(P.sum(0) - w_tgt).sum()
        
        info = {
            'cost_time': cost_time,
            'sinkhorn_time': sinkhorn_time,
            'total_time': time.time() - start_time,
            'marginal_error_1': err1,
            'marginal_error_2': err2,
            'regularization': reg,
            'method': method
        }
        
        return P, info
    
    # Test reference implementation
    print("Testing POT reference implementation...")
    P_ref, info_ref = pot_reference_ot(X_src, X_tgt, w_src, w_tgt, reg=0.01)
    
    print(f"✅ POT Reference Results:")
    print(f"   Transport plan shape: {P_ref.shape}")
    print(f"   Total time: {info_ref['total_time']:.2f}s")
    print(f"   Marginal errors: {info_ref['marginal_error_1']:.2e}, {info_ref['marginal_error_2']:.2e}")
    print(f"   Plan sum: {P_ref.sum():.6f} (should be ≈1.0)")
    print(f"   Plan range: [{P_ref.min():.2e}, {P_ref.max():.2e}]")
    
    # Save reference results
    np.savez('pot_reference.npz', 
             P_reference=P_ref, 
             info=info_ref,
             C_reference=ot.dist(X_src, X_tgt, metric='sqeuclidean'))
    print("✅ POT reference saved to pot_reference.npz")
    
    return pot_reference_ot

if __name__ == "__main__":
    print("=== POT Setup and Exploration ===")
    
    # Step 1: Test installation
    if not test_pot_installation():
        print("❌ Cannot proceed without POT installation")
        exit(1)
    
    # Step 2: Explore Sinkhorn variants
    explore_sinkhorn_variants()
    
    # Step 3: Test with RGB data
    C, results, reg_values = test_sinkhorn_log_stabilized()
    
    # Step 4: Analyze API
    analyze_pot_api()
    
    # Step 5: Create reference implementation
    pot_reference_ot = create_pot_reference()
    
    print("\n=== POT Setup Complete ===")
    print("✅ POT library verified and tested")
    print("✅ Log-stabilized Sinkhorn functions identified") 
    print("✅ RGB data tested with POT algorithms")
    print("✅ Reference implementation created for validation")
    print("✅ Ready for Step 4: Implementing our own cost matrix computation")
    print("\n📋 Key POT Functions for Our Implementation:")
    print("   - ot.dist(X1, X2, metric='sqeuclidean') → cost matrix")
    print("   - ot.sinkhorn(a, b, C, reg) → transport plan (standard)")
    print("   - ot.sinkhorn_stabilized(a, b, C, reg) → transport plan (stabilized)")
    print("   - Input: source/target weights (a, b) and cost matrix (C)")
    print("   - Output: transport plan P where P.sum(1) ≈ a, P.sum(0) ≈ b")
