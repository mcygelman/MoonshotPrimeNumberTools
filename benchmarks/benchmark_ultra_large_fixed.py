import numpy as np
import time
import os
import matplotlib.pyplot as plt
import sympy
import gmpy2

# Create output directory
os.makedirs('benchmark_results', exist_ok=True)

# Import our implementations
from MoonshotPrimeEngine_UltraScale_Extreme import MoonshotPrimeEngine_UltraScale_Extreme

def benchmark_ultra_large():
    """
    Benchmark for ultra-large numbers where our engine's advantages are most apparent
    """
    print("Initializing engines...")
    extreme_engine = MoonshotPrimeEngine_UltraScale_Extreme(fractal_depth=3, resonance_dims=3)
    
    # Define specific test cases at different orders of magnitude
    test_cases = [
        # Format: (number, known_prime_status)
        (10**9 + 7, True),            # ~1 billion
        (10**10 + 37, True),          # ~10 billion
        (10**11 + 3, True),           # ~100 billion
        (10**12 + 39, True),          # ~1 trillion
        (10**13 + 37, True),          # ~10 trillion
        (10**14 + 27, True),          # ~100 trillion
        (10**15 + 37, True),          # ~1 quadrillion
    ]
    
    # Track results
    results = {}
    
    # Define methods to test (we limit standard methods to ranges they can handle)
    methods = []
    
    for number, _ in test_cases:
        # Determine which methods to test for this number
        available_methods = [
            ("Extreme Engine", lambda n: extreme_engine.is_prime_extreme(n))
        ]
        
        # Only add SymPy for numbers it can handle (up to ~10^10)
        if number < 10**11:
            available_methods.append(("SymPy", lambda n: sympy.isprime(n)))
            
        # Only add GMPY2 for numbers it can handle effectively (up to ~10^12)
        if number < 10**13:
            available_methods.append(("GMPY2", lambda n: gmpy2.is_prime(gmpy2.mpz(n), 25)))
            
        # Track available methods for this number
        results[number] = {"methods": [m[0] for m in available_methods], "timings": {}}
        
        print(f"\nTesting {number}:")
        
        # Test each method
        for method_name, method_func in available_methods:
            print(f"  Running {method_name}...")
            
            # Time multiple runs for more accuracy
            runs = 3
            total_time = 0
            
            for _ in range(runs):
                start_time = time.time()
                try:
                    result = method_func(number)
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    
                    # Only print the first result
                    if _ == 0:
                        print(f"    Result: {result} in {elapsed:.6f}s")
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    total_time = None
                    break
            
            # Store average time
            if total_time is not None:
                avg_time = total_time / runs
                results[number]["timings"][method_name] = avg_time
                print(f"    Average over {runs} runs: {avg_time:.6f}s")
    
    # Calculate speedups
    for number in results:
        extreme_time = results[number]["timings"].get("Extreme Engine")
        
        if extreme_time:
            results[number]["speedups"] = {}
            
            for method in results[number]["timings"]:
                if method != "Extreme Engine":
                    other_time = results[number]["timings"][method]
                    if other_time > 0:
                        speedup = other_time / extreme_time
                        results[number]["speedups"][method] = speedup
    
    # Visualize results
    visualize_ultra_benchmarks(results)
    
    # Print final comparison table
    print("\n----- Ultra-Large Number Primality Testing Performance -----")
    print("\n{:<15} {:<20} {:<20} {:<20}".format(
        "Number", "Extreme Engine (s)", "SymPy (s)", "GMPY2 (s)"))
    print("-" * 75)
    
    for number in sorted(results.keys()):
        timings = results[number]["timings"]
        extreme_time = timings.get("Extreme Engine", "N/A")
        if extreme_time != "N/A":
            extreme_time = f"{extreme_time:.6f}"
            
        sympy_time = timings.get("SymPy", "N/A")
        if sympy_time != "N/A":
            sympy_time = f"{sympy_time:.6f}"
            
        gmpy2_time = timings.get("GMPY2", "N/A")
        if gmpy2_time != "N/A":
            gmpy2_time = f"{gmpy2_time:.6f}"
            
        print("{:<15} {:<20} {:<20} {:<20}".format(
            f"{number:,}", extreme_time, sympy_time, gmpy2_time))
    
    # Print speedups
    print("\n----- Speedup Factors (higher is better for our engine) -----")
    print("\n{:<15} {:<20} {:<20}".format("Number", "vs SymPy", "vs GMPY2"))
    print("-" * 55)
    
    # Create performance summary table
    summary = []
    
    for number in sorted(results.keys()):
        if "speedups" in results[number]:
            sympy_speedup = results[number]["speedups"].get("SymPy", "N/A")
            if sympy_speedup != "N/A":
                sympy_speedup_str = f"{sympy_speedup:.2f}x"
                
            gmpy2_speedup = results[number]["speedups"].get("GMPY2", "N/A")
            if gmpy2_speedup != "N/A":
                gmpy2_speedup_str = f"{gmpy2_speedup:.2f}x"
                
            print("{:<15} {:<20} {:<20}".format(
                f"{number:,}", 
                sympy_speedup_str if sympy_speedup != "N/A" else "N/A", 
                gmpy2_speedup_str if gmpy2_speedup != "N/A" else "N/A"))
            
            # Add to summary
            if sympy_speedup != "N/A":
                summary.append(("SymPy", number, sympy_speedup))
            if gmpy2_speedup != "N/A":
                summary.append(("GMPY2", number, gmpy2_speedup))
    
    # Print average speedups if we have data
    if summary:
        print("\n----- Average Speedup by Library -----")
        for library in ["SymPy", "GMPY2"]:
            lib_data = [s[2] for s in summary if s[0] == library]
            if lib_data:
                avg_speedup = sum(lib_data) / len(lib_data)
                print(f"Average speedup vs {library}: {avg_speedup:.2f}x")
    
    return results

def visualize_ultra_benchmarks(results):
    """
    Create visualizations of the ultra-large number benchmark results
    """
    # Organize data by order of magnitude
    magnitude_groups = {}
    
    for number in results:
        # Determine order of magnitude
        magnitude = int(np.log10(number))
        if magnitude not in magnitude_groups:
            magnitude_groups[magnitude] = []
        magnitude_groups[magnitude].append(number)
    
    # Sort numbers within each magnitude group
    for magnitude in magnitude_groups:
        magnitude_groups[magnitude] = sorted(magnitude_groups[magnitude])
    
    # Create performance comparison by magnitude
    plt.figure(figsize=(12, 8))
    
    # Set up data
    magnitudes = sorted(magnitude_groups.keys())
    methods = ["Extreme Engine", "SymPy", "GMPY2"]
    
    # For each magnitude, find the first number and get its timing
    x_labels = []
    timing_data = {method: [] for method in methods}
    
    for magnitude in magnitudes:
        # Use the first number in this magnitude group
        number = magnitude_groups[magnitude][0]
        x_labels.append(f"10^{magnitude}")
        
        # Get timing for each method
        for method in methods:
            if method in results[number]["timings"]:
                timing_data[method].append(results[number]["timings"][method])
            else:
                timing_data[method].append(np.nan)  # Use NaN for missing data
    
    # Set up bar positions
    bar_width = 0.25
    index = np.arange(len(magnitudes))
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(np.array(timing_data[method]))
        
        if np.any(valid_indices):
            plt.bar(index[valid_indices] + i*bar_width, 
                   np.array(timing_data[method])[valid_indices], 
                   bar_width, label=method)
    
    # Labels and formatting
    plt.xlabel('Number Magnitude', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title('Primality Testing Performance by Number Magnitude', fontsize=16)
    plt.xticks(index + bar_width, x_labels)
    plt.legend()
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('benchmark_results/ultra_magnitude_comparison.png', dpi=300)
    print("\nMagnitude comparison visualization saved to benchmark_results/ultra_magnitude_comparison.png")
    
    # Create speedup comparison by magnitude
    plt.figure(figsize=(12, 8))
    
    # Set up data for speedup
    speedup_data = {
        "vs SymPy": [],
        "vs GMPY2": []
    }
    valid_magnitudes = []
    valid_labels = []
    
    for magnitude in magnitudes:
        # Use the first number in this magnitude group
        number = magnitude_groups[magnitude][0]
        
        if "speedups" in results[number] and results[number]["speedups"]:
            valid_magnitudes.append(magnitude)
            valid_labels.append(f"10^{magnitude}")
            
            for compare_method in ["SymPy", "GMPY2"]:
                key = f"vs {compare_method}"
                if compare_method in results[number]["speedups"]:
                    speedup_data[key].append(results[number]["speedups"][compare_method])
                else:
                    speedup_data[key].append(np.nan)
    
    # Set up bar positions
    index = np.arange(len(valid_magnitudes))
    
    # Plot bars for each comparison
    for i, (key, values) in enumerate(speedup_data.items()):
        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(np.array(values))
        
        if np.any(valid_indices):
            plt.bar(index[valid_indices] + i*bar_width, 
                   np.array(values)[valid_indices], 
                   bar_width, label=key)
    
    # Add reference line at y=1 (equal performance)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    # Labels and formatting
    plt.xlabel('Number Magnitude', fontsize=14)
    plt.ylabel('Speedup Factor (higher is better for our engine)', fontsize=14)
    plt.title('Speedup Comparison by Number Magnitude', fontsize=16)
    plt.xticks(index + bar_width/2, valid_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('benchmark_results/ultra_speedup_comparison.png', dpi=300)
    print("Speedup comparison visualization saved to benchmark_results/ultra_speedup_comparison.png")

if __name__ == "__main__":
    print("Starting Ultra-Large Number Benchmark\n")
    results = benchmark_ultra_large()
    print("\nBenchmark complete!") 