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

def benchmark_primality_tests():
    """
    Benchmark our implementation against standard libraries
    """
    print("Initializing engines...")
    extreme_engine = MoonshotPrimeEngine_UltraScale_Extreme(fractal_depth=3, resonance_dims=3)
    
    # Test ranges - using logarithmically increasing ranges
    ranges = [
        (1000, 10000),           # Thousands
        (100000, 1000000),       # Hundred thousands
        (10000000, 100000000),   # Tens of millions
        (1000000000, 10000000000), # Billions
        (10**11, 10**12),        # Hundred billions
        (10**13, 10**14)         # Ten trillions
    ]
    
    # Define a set of known primes for accurate testing
    known_primes = [
        7919,              # ~8K
        104729,            # ~100K
        15485863,          # ~15M
        2038074743,        # ~2B
        1000000000039,     # ~1T
        10000000000037     # ~10T
    ]
    
    # Testing non-primes
    non_primes = [
        7918,              # ~8K
        104730,            # ~100K
        15485864,          # ~15M
        2038074744,        # ~2B
        1000000000038,     # ~1T
        10000000000038     # ~10T
    ]
    
    all_test_numbers = known_primes + non_primes
    
    # Set up results storage
    results = {num: {} for num in all_test_numbers}
    
    # Test each primality method
    methods = [
        ("Extreme Engine", lambda n: extreme_engine.is_prime_extreme(n)),
        ("SymPy", lambda n: sympy.isprime(n)),
        ("GMPY2", lambda n: gmpy2.is_prime(gmpy2.mpz(n), 25))  # Fixed GMPY2 with proper conversion and certainty level
    ]
    
    print("\nTesting specific numbers with all methods:")
    for num in all_test_numbers:
        print(f"\nTesting {num}:")
        
        for method_name, method_func in methods:
            # Skip extremely large numbers for standard libraries if needed
            if num > 10**10 and method_name == "SymPy":
                results[num][method_name] = {"time": None, "result": None}
                print(f"  {method_name}: Skipped (too large)")
                continue
                
            if num > 10**12 and method_name == "GMPY2":
                results[num][method_name] = {"time": None, "result": None}
                print(f"  {method_name}: Skipped (too large)")
                continue
                
            # Measure time
            start_time = time.time()
            try:
                result = method_func(num)
                elapsed = time.time() - start_time
                
                # Store result
                results[num][method_name] = {
                    "time": elapsed,
                    "result": result
                }
                
                # Print result
                print(f"  {method_name}: {result} in {elapsed:.6f}s")
            except Exception as e:
                print(f"  {method_name}: Error - {str(e)}")
                results[num][method_name] = {"time": None, "result": None}
    
    # Range benchmark
    range_results = {}
    for start, end in ranges:
        range_name = f"{start}-{end}"
        print(f"\nBenchmarking range {range_name}...")
        
        # Generate sample numbers (fewer for demonstration)
        sample_size = 3
        log_start, log_end = np.log10(start), np.log10(end)
        log_samples = np.linspace(log_start, log_end, sample_size)
        numbers = np.power(10, log_samples).astype(int)
        
        range_results[range_name] = {}
        
        for method_name, method_func in methods:
            # Skip large ranges for standard libraries
            if end > 10**10 and method_name == "SymPy":
                range_results[range_name][method_name] = {"avg_time": None}
                print(f"  {method_name}: Skipped (range too large)")
                continue
                
            if end > 10**12 and method_name == "GMPY2":
                range_results[range_name][method_name] = {"avg_time": None}
                print(f"  {method_name}: Skipped (range too large)")
                continue
                
            # Test performance
            total_time = 0
            successful_tests = 0
            
            for n in numbers:
                try:
                    start_time = time.time()
                    if method_name == "GMPY2":
                        _ = method_func(n)  # Properly convert to mpz for GMPY2
                    else:
                        _ = method_func(n)
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    successful_tests += 1
                except Exception as e:
                    print(f"  Error testing {n} with {method_name}: {str(e)}")
            
            # Calculate average
            if successful_tests > 0:
                avg_time = total_time / successful_tests
                range_results[range_name][method_name] = {"avg_time": avg_time}
                print(f"  {method_name}: {avg_time:.6f}s/number")
            else:
                range_results[range_name][method_name] = {"avg_time": None}
                print(f"  {method_name}: No successful tests")
    
    # Create visualizations
    visualize_comparison(results, range_results)
    
    return results, range_results

def visualize_comparison(results, range_results):
    """Create visualizations of benchmark results"""
    
    # Prepare data for specific numbers plot
    numbers = sorted(results.keys())
    methods = ["Extreme Engine", "SymPy", "GMPY2"]
    
    # Filter to known primes for clearer visualization
    prime_numbers = [n for n in numbers if n in [7919, 104729, 15485863, 2038074743, 1000000000039, 10000000000037]]
    
    # Group into size categories for clearer visualization
    size_categories = {
        "Small (thousands)": [7919],
        "Medium (hundred thousands)": [104729],
        "Large (millions)": [15485863],
        "Very Large (billions)": [2038074743],
        "Ultra Large (trillions)": [1000000000039, 10000000000037]
    }
    
    # Plot for each size category
    for category_name, category_numbers in size_categories.items():
        plt.figure(figsize=(10, 6))
        
        # Extract timing data
        category_times = {}
        valid_methods = []
        
        for method in methods:
            method_times = []
            has_valid_data = False
            
            for num in category_numbers:
                if num in results and method in results[num] and results[num][method]["time"] is not None:
                    method_times.append(results[num][method]["time"])
                    has_valid_data = True
                else:
                    method_times.append(0)  # Use 0 for missing data
            
            if has_valid_data:
                category_times[method] = method_times
                valid_methods.append(method)
        
        # Set up bar positions
        bar_width = 0.25
        index = np.arange(len(category_numbers))
        
        # Plot bars for each method
        for i, method in enumerate(valid_methods):
            plt.bar(index + i*bar_width, category_times[method], bar_width, label=method)
        
        # Labels and formatting
        plt.xlabel('Prime Number')
        plt.ylabel('Time (seconds)')
        plt.title(f'Primality Testing Performance - {category_name}')
        plt.xticks(index + bar_width, [str(n) for n in category_numbers])
        plt.legend()
        plt.yscale('log')  # Log scale for better visibility
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'benchmark_results/comparison_{category_name.replace(" ", "_").lower()}.png', dpi=300)
        print(f"Comparison visualization saved for {category_name}")
    
    # Prepare range data
    range_names = sorted(range_results.keys())
    
    # Split ranges into groups for clearer visualization
    range_groups = {
        "Small to Medium Ranges": range_names[:3],
        "Large to Ultra Ranges": range_names[3:]
    }
    
    # Plot for each range group
    for group_name, group_ranges in range_groups.items():
        plt.figure(figsize=(12, 8))
        
        # Extract timing data for ranges
        range_times = {}
        valid_methods = []
        
        for method in methods:
            method_times = []
            has_valid_data = False
            
            for r in group_ranges:
                if r in range_results and method in range_results[r] and range_results[r][method]["avg_time"] is not None:
                    method_times.append(range_results[r][method]["avg_time"])
                    has_valid_data = True
                else:
                    method_times.append(0)  # Use 0 for missing data
            
            if has_valid_data:
                range_times[method] = method_times
                valid_methods.append(method)
        
        # Set up bar positions
        bar_width = 0.25
        index = np.arange(len(group_ranges))
        
        # Plot bars for each method
        for i, method in enumerate(valid_methods):
            plt.bar(index + i*bar_width, range_times[method], bar_width, label=method)
        
        # Labels and formatting
        plt.xlabel('Number Range')
        plt.ylabel('Average Time (seconds)')
        plt.title(f'Average Primality Testing Performance - {group_name}')
        plt.xticks(index + bar_width, group_ranges, rotation=45)
        plt.legend()
        plt.yscale('log')  # Log scale for better visibility
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'benchmark_results/range_comparison_{group_name.replace(" ", "_").lower()}.png', dpi=300)
        print(f"Range comparison visualization saved for {group_name}")
    
    # Calculate and display speedup for each prime in a comprehensive table
    print("\nSpeedup Summary for Individual Primes:")
    print("\n{:<15} {:<25} {:<25} {:<25}".format("Number", "Extreme Time (s)", "vs SymPy", "vs GMPY2"))
    print("-" * 90)
    
    for num in sorted(results.keys()):
        if num in prime_numbers:  # Only show primes
            extreme_data = results[num].get("Extreme Engine", {})
            extreme_time = extreme_data.get("time")
            
            if extreme_time:
                sympy_time = results[num].get("SymPy", {}).get("time")
                gmpy2_time = results[num].get("GMPY2", {}).get("time")
                
                sympy_speedup = sympy_time / extreme_time if sympy_time else "N/A"
                gmpy2_speedup = gmpy2_time / extreme_time if gmpy2_time else "N/A"
                
                sympy_str = f"{sympy_speedup:.2f}x" if isinstance(sympy_speedup, float) else sympy_speedup
                gmpy2_str = f"{gmpy2_speedup:.2f}x" if isinstance(gmpy2_speedup, float) else gmpy2_speedup
                
                print("{:<15} {:<25.6f} {:<25} {:<25}".format(
                    str(num), extreme_time, sympy_str, gmpy2_str))
    
    # Calculate and display speedup for ranges
    print("\nSpeedup Summary for Ranges:")
    print("\n{:<20} {:<25} {:<25} {:<25}".format("Range", "Extreme Time (s)", "vs SymPy", "vs GMPY2"))
    print("-" * 95)
    
    for r in range_names:
        if r in range_results:
            extreme_data = range_results[r].get("Extreme Engine", {})
            extreme_time = extreme_data.get("avg_time")
            
            if extreme_time:
                sympy_time = range_results[r].get("SymPy", {}).get("avg_time")
                gmpy2_time = range_results[r].get("GMPY2", {}).get("avg_time")
                
                sympy_speedup = sympy_time / extreme_time if sympy_time else "N/A"
                gmpy2_speedup = gmpy2_time / extreme_time if gmpy2_time else "N/A"
                
                sympy_str = f"{sympy_speedup:.2f}x" if isinstance(sympy_speedup, float) else sympy_speedup
                gmpy2_str = f"{gmpy2_speedup:.2f}x" if isinstance(gmpy2_speedup, float) else gmpy2_speedup
                
                print("{:<20} {:<25.6f} {:<25} {:<25}".format(
                    r, extreme_time, sympy_str, gmpy2_str))

if __name__ == "__main__":
    print("Starting Primality Test Benchmark Comparison\n")
    results, range_results = benchmark_primality_tests()
    print("\nBenchmark complete!") 